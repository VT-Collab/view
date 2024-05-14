import json
import os
import math
import gym
import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pybullet as p
# import pybullet as p
import pybullet_data
from gym import spaces
from gym_panda.envs.objects import YCBObject, RBOObject
from gym_panda.envs.robot_control import Panda
import time

YCB_Objects = os.listdir("assets/ycb")
RBO_Objects = os.listdir("assets/rbo")

class PandaEnv(gym.Env):
    def __init__(self, args, max_t=50000):


        self.action_space = spaces.Box(low=-1.0,
                                       high=+1.0, 
                                       shape=(8,), # cartesian position, orientation in quaternion, gripper state
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=+np.inf, 
                                            shape=(8,), # cartesian position, orientation in quaternion, gripper state
                                            dtype=np.float64)

        # sim params
        self.timesteps = 0
        self.max_t = max_t
        # create simulation (GUI)
        self.urdfRootPath = pybullet_data.getDataPath()
        if args.gui:
            p.connect(p.GUI, options="--width=2000 --height=2000")
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(args.dt)

        # set up camera
        self._set_camera()

        # load some scene objects
        # p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"))
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, 0.])
        self.object_names = args.objects
        self.object_positions = args.object_positions
        self.n_objects = len(args.objects)
        self.objects = []

        for idx, object_name in enumerate(args.objects):
            # load object
            if object_name in YCB_Objects:
                obj = YCBObject(object_name)
            elif object_name in RBO_Objects:
                obj = RBOObject(object_name)
            else:
                raise ValueError("Unknown object {}".format(object_name))
            
            obj.load()
            p.resetBasePositionAndOrientation(obj.body_id, self.object_positions[idx, :3], self.object_positions[idx, 3:])
            self.objects.append(obj)

        # load a panda robot
        self.grasp = 0
        self.robot_home = [0.020, -0.91, -0.01, -2.33, -0.00, 1.41, 0.80, 0.05, 0.05]
        self.panda = Panda([0,0,0.63])
        self.reset()

    def _get_obs(self):
        state = np.array(self.panda.state["ee_position"].tolist() + self.panda.state["ee_quaternion"].tolist() + [self.grasp])
        return state

    def reset(self, seed=None, q=None):
        super().reset(seed=seed)
        if q is None:
            q = self.robot_home
        self.panda.reset(q=q)
        self.panda.reset_gripper()
        self.grasp = 0
        
        for idx, obj in enumerate(self.objects):
            p.resetBasePositionAndOrientation(obj.body_id, self.object_positions[idx, :3], self.object_positions[idx, 3:])
        
        self.timesteps = 0
        return self._get_obs()

    def reward(self):
        return -1.

    def step(self, action):
        # tic = time.time()
        self.timesteps += 1

        self.grasp = int(np.round(action[-1]))

        self.panda.step(mode=1, dposition=action[:3], dquaternion=action[3:7], grasp_open=1-self.grasp)
        p.stepSimulation()
        img = None
        # img = self.render()

        done = False
        if self.timesteps == self.spec.max_episode_steps or self.timesteps == self.max_t:
            done = True
        truncated = False

        object_positions = dict()
        for idx, obj in enumerate(self.objects):
            pos_orien = obj.get_position_orientation()
            object_positions[self.object_names[idx]] = pos_orien

        # print(time.time()-tic)
        return self._get_obs(), self.reward(), done, truncated, {"object_positions":object_positions, "img":img}

    def get_inverse_kinematics(self, ee_position, ee_quaternion):
        return self.panda._inverse_kinematics(ee_position, ee_quaternion)

    def close(self):
        p.disconnect()

    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
        )
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(
            cameraDistance=1.4,
            cameraYaw=20.80,
            cameraPitch=-40.00,
            cameraTargetPosition=[0.57, -0.0, 0.57],
        )
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.57, -0.00, 0.79],
            distance=1.4,
            yaw=19.6,
            pitch=-49.2,
            roll=0,
            upAxisIndex=2,
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.camera_width) / self.camera_height,
            nearVal=0.1,
            farVal=100.0,
        )