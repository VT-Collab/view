import os
import json
import gym
import gym_panda
import numpy as np
import argparse
import yaml
import cv2
import matplotlib.pyplot as plt
from utils import Trajectory
import warnings
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--name", required=True, help="Name of demonstration")
parser.add_argument("--dt", default=1e-4, type=float, help="Number of points to keep after compressing")
args = parser.parse_args()

cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
savefolder = os.path.join(cfg["save_data"]["DEMOS"], args.name)
imgfolder = os.path.join(savefolder, "imgs")

if not os.path.exists(imgfolder):
    os.makedirs(imgfolder)

# Sim related arguments
args.objects = ["025_mug"]
args.object_positions = np.array([[0.4, -0.3, 0.68, 0.0, 0.0, 1.0, 0.0]])
sim = gym.make('panda-v0', args=args)

# robot_home = [0.020, -0.91, -0.01, -2.33, -0.00, 2.81, -0.80, 0.05, 0.05]
robot_state = sim.reset()
initial_pos = robot_state[:3].copy()
initial_ang = robot_state[3:7].copy()
initial_grip = robot_state[-1].copy()

obj_position = args.object_positions[0, :3]
start = initial_pos.tolist() +  [0]
pickup = [0.4, -0.35, 0.7, 1]
intermediate = [0.45, -0.1, 1.15, 1]
place = [0.5, 0.1, 0.7, 0]
# end = [0.5, 0.,1.15, 0]

waypoints = np.array([start, pickup, intermediate, place])
times = np.arange(len(waypoints)) * 2.0
traj = np.column_stack(((times, waypoints)))
traj_fn = Trajectory((traj))

fig, axs = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(15,15))
axs.plot(traj[:, 1], traj[:, 2], traj[:, 3], 'ko', label='Given Waypoints')

t = 0
dt = args.dt
img_num = 0
max_t = times[-1] + 2.0 # give two extra seconds for sim 
final_state = waypoints[-1,:]
object_of_interest = args.objects[0]

demo_traj = []
obj_traj = []
while t < max_t:
    target = traj_fn.get_waypoint(t) # contains [x, y, z, gripper]
    linear = target[:3]# - robot_state[:3]
    angular = initial_ang.copy() # no orientation in traj fn
    grip = target[-1]
    action = np.hstack((linear, angular, [grip]))
    robot_state, _,_, info = sim.step(action)
    state = np.hstack((robot_state[:3], robot_state[-1]))
    object_state = np.array(info['object_positions'][object_of_interest][:3]).squeeze() # remove orientation of object

    if not np.round(t * 1/dt) % 500: # post every second
        # img = cv2.cvtColor(info["img"], cv2.COLOR_RGB2BGR)
        # img_name = "img_" + str(img_num) + ".jpg"
        # cv2.imwrite(os.path.join(imgfolder, img_name), img)
        print("t: {}\nstate: {}\nobj_state: {}\naction: {}\n----".format(t, state, object_state, action))
        input()

    if np.linalg.norm(final_state-state) < 0.01:
        # json.dump(waypoints.tolist(), open(os.path.join(savefolder, "original_waypoints.json"), "w"))
        # json.dump(demo_traj, open(os.path.join(savefolder, "traj.json"), "w"))
        # json.dump(obj_traj, open(os.path.join(savefolder, "obj_traj.json"), "w"))

        demo_traj = np.array(demo_traj)
        obj_traj = np.array(obj_traj)
        axs.plot(demo_traj[:, 1], demo_traj[:, 2], demo_traj[:, 3], 'b', label='Robot Traj')
        axs.plot(obj_traj[:, 1], obj_traj[:, 2], obj_traj[:, 3], 'r', label=f'{object_of_interest} traj')
        axs.legend()
        plt.tight_layout()
        # plt.savefig(os.path.join(savefolder, "result.jpg"))
        break
    
    # save relevant info
    if not np.round(t * 1/dt) % 500:
        demo_traj.append([t] + state.tolist())
        obj_traj.append([t] + object_state.tolist())

    # step
    t += dt 
    img_num += 1

# show resulting trajectories
plt.show()