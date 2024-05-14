import os
import json
import gym
import gym_panda
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import Trajectory, ManipulateReward
import yaml
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--name", required=True, help="Name of demonstration")
args = parser.parse_args()

cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

demofolder = os.path.join(cfg["save_data"]["DEMOS"], args.name)
dir = args.name + time.strftime('_%Y%m%d_%H%M%S')
rolloutfolder = os.path.join(cfg["save_data"]["ROLLOUTS"], dir)

if not os.path.exists(rolloutfolder):
    os.makedirs(rolloutfolder)

# Sim related arguments
args.objects = ["025_mug"]
args.object_positions = np.array([[0.4, -0.3, 0.65, 0.0, 0.0, 1.0, 0.0]])
sim = gym.make('panda-v0', args=args)

robot_state = sim.reset()
initial_pos = robot_state[:3].copy()
initial_ang = robot_state[3:7].copy()
initial_grip = robot_state[-1].copy()

obj_position = args.object_positions[0, :3]
start = initial_pos.tolist() +  [0]
pickup = obj_position.tolist() + [1]
# pickup[2] = 0.95 # distort prior
end = [0.5, -0.2, 1.15, 1]

times = [0., 1., 2.]
waypoints = np.array([start, pickup, end])
traj = np.column_stack(((times, waypoints)))
traj_fn = Trajectory((traj))
reward_fn = ManipulateReward(demofolder)

object_of_interest = args.objects[0]
fig, axs = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(15,15))
demo_obj_traj = np.array(json.load(open(os.path.join(demofolder, "obj_traj.json"), "r")))
axs.plot(demo_obj_traj[:, 1], demo_obj_traj[:, 2], demo_obj_traj[:, 3], 'k', label=f'{object_of_interest} demo traj')

t = 0
dt = 1e-4 # DO NOT CHANGE
max_t = times[-1] + 2.0 # give two extra seconds for sim 
final_state = waypoints[-1,:]

demo_traj = []
obj_traj = []
while True:
    target = traj_fn.get_waypoint(t) # contains [x, y, z, gripper]
    linear = target[:3] - robot_state[:3]
    angular = np.zeros(4) # no orientation in traj fn
    grip = target[-1]
    action = np.hstack((linear, angular, [grip]))
    
    robot_state, _,_, info = sim.step(action)
    state = np.hstack((robot_state[:3], robot_state[-1]))
    object_state = np.array(info['object_positions'][object_of_interest][:3]) # remove orientation of object

    if not np.round(t * 1/dt) % 1000: # post every second
        print("t: {}\nstate: {}\nobj_state: {}\naction: {}\n----".format(t, state, object_state, action))

    if np.linalg.norm(final_state-state) < 0.015 or t > max_t:
        
        json.dump(demo_traj, open(os.path.join(rolloutfolder, "traj.json"), "w"))
        json.dump(obj_traj, open(os.path.join(rolloutfolder, "obj_traj.json"), "w"))

        demo_traj = np.array(demo_traj)
        obj_traj = np.array(obj_traj)
        
        relevant_times = [times[-1]]
        reward = reward_fn.compute_reward(obj_traj, relevant_times)

        print("Reward = {}".format(reward))        
        axs.plot(demo_traj[:, 1], demo_traj[:, 2], demo_traj[:, 3], 'b', label='Robot Traj')
        axs.plot(obj_traj[:, 1], obj_traj[:, 2], obj_traj[:, 3], 'r', label=f'{object_of_interest} rollout traj')
        axs.legend()
        axs.set_title("Reward = {}".format(np.round(reward, 3)))
        plt.tight_layout()
        plt.savefig(os.path.join(rolloutfolder, "result.jpg"))
        break
    
    # save relevant info
    demo_traj.append([t] + state.tolist())
    obj_traj.append([t] + object_state.tolist())

    # step
    t += dt 

# show resulting trajectories
plt.show()