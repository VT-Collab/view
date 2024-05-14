import gym
import gym_panda
import numpy as np
import argparse
import time
from utils import Trajectory

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.objects = ["025_mug", "011_banana"]
args.object_positions = np.array([[0.4, -0.3, 0.65, 0.0, 0.0, 1.0, 0.0], [0.5, -0.3, 0.03, 0.0, 0.0, 0.0, 1.0]])
args.gui = True
sim = gym.make('panda-v0', args=args)

robot_state = sim.reset()
initial_pos = robot_state[:3].copy()
initial_ang = robot_state[3:7].copy()
initial_grip = robot_state[-1].copy()

obj_position = args.object_positions[0, :3]
start = initial_pos.tolist() +  [0]
pickup = obj_position.tolist() + [1]
intermediate = [0.45, 0., 1.05, 1]
place = [0.5, 0.1, 0.7, 0]
end = [0.5, 0.2, 1.15, 1]

waypoints = np.array([start, pickup, end])
times = np.arange(len(waypoints))
traj = np.column_stack(((times, waypoints)))
traj_fn = Trajectory((traj))

t = 0
dt = 1/240.
max_t = times[-1] + 2.0 # give two extra seconds for sim 

while True:
    target = traj_fn.get_waypoint(t) # contains [x, y, z, gripper]
    angular = initial_ang.copy() # no orientation in traj fn
    grip = target[-1]
    action = np.hstack((target[:3], angular, [grip]))
    
    robot_state, _,_, info = sim.step(action)
    state = np.hstack((robot_state[:3], robot_state[-1]))

    if not np.round(t * 1/dt) % 1000: # post every second
        print("t: {}\nstate: {}\naction: {}\n----".format(t, state, action))
    # step
    t += dt 
