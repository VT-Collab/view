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

def generate_traj(traj_type, lower_limits, upper_limits):
    assert traj_type in ["pick", "move", "push"], "Undefined traj type."
    # we ignore height for object placement
    obj_xy_position = np.random.uniform(lower_limits[:2], upper_limits[:2], (1,2))
    obj_xy_position = obj_xy_position.squeeze().tolist()

    obj_position = obj_xy_position + [0.68]

    start = [0.254, 0.002, 1.148, 0]
    pickup = obj_position + [1]
    pickup[1] += 0.05 # offset by 5cm to actually pickup the object

    if traj_type == "pick":
        # set a height range above the table
        lower_limits[2] = 0.75
        upper_limits[2] = 0.9
        end = np.random.uniform(lower_limits[:3], upper_limits[:3], (1,3)).squeeze().tolist()
        end = end + [1] # add gripper state
        waypoints = [start, pickup, end]
    elif traj_type == "push":
        place = np.random.uniform(lower_limits[:3], upper_limits[:3], (1,3)).squeeze().tolist()
        place[2] = 0.68 # fixed height from table
        place = place + [0]
        end = place[:2] + [1.1, 0]
        waypoints = [start, pickup, place, end]
    else:
        place = np.random.uniform(lower_limits[:3], upper_limits[:3], (1,3)).squeeze().tolist()
        place[2] = 0.68 # fixed height from table
        place = place + [0]
        intermediate = [np.mean([pickup[0], place[0]]), np.mean([pickup[1], place[1]]), 0.9, 1]
        end = place[:2] + [1.1, 0]
        waypoints = [start, pickup, intermediate, place, end]
    waypoints = np.array(waypoints)
    times = np.arange(len(waypoints)) * 2.0
    traj = np.column_stack(((times, waypoints)))
    return traj, obj_position

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--rollouts", type=int, default=50, help="Number of rollouts for each type of trajectory")
    parser.add_argument("--dt", default=1e-4, type=float, help="Number of points to keep after compressing")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    limit_x = [0.15, 0.6]
    limit_y = [-0.4, 0.4]
    limit_z = [0.68, 1.2]
    
    lower_limits = [limit_x[0], limit_y[0], limit_z[0]]
    upper_limits = [limit_x[1], limit_y[1], limit_z[1]]

    traj_types = ["pick", "push", "move"]
    for traj_type in traj_types:
        for roll_num in range(args.rollouts):
            name = traj_type + "_" + str(roll_num) 
            savefolder = os.path.join(cfg["save_data"]["DEMOS"], name)
            imgfolder = os.path.join(savefolder, "imgs")

            if not os.path.exists(imgfolder):
                os.makedirs(imgfolder)

            traj, object_position = generate_traj(traj_type, lower_limits, upper_limits)
            traj_fn = Trajectory((traj))
            times = traj[:,0]
            waypoints = traj[:,1:]

            fig, axs = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(15,15))
            axs.plot(traj[:, 1], traj[:, 2], traj[:, 3], 'ko', label='Given Waypoints')

            t = 0
            dt = args.dt
            img_num = 0
            max_t = times[-1] + 2.0 # give two extra seconds for sim 
            final_state = waypoints[-1,:]

            demo_traj = []
            obj_traj = []

            # Sim related arguments
            args.objects = ["025_mug"]
            args.object_positions = np.array([object_position + [0.0, 0.0, 1.0, 0.0]])
            object_of_interest = args.objects[0]


            sim = gym.make('panda-v0', args=args)
            robot_state = sim.reset()
            initial_pos = robot_state[:3].copy()
            initial_ang = robot_state[3:7].copy()
            initial_grip = robot_state[-1].copy()

            while t < max_t:
                target = traj_fn.get_waypoint(t) # contains [x, y, z, gripper]
                linear = target[:3]# - robot_state[:3]
                angular = initial_ang.copy() # no orientation in traj fn
                grip = target[-1]
                action = np.hstack((linear, angular, [grip]))
                
                robot_state, _,_, info = sim.step(action)
                state = np.hstack((robot_state[:3], robot_state[-1]))
                object_state = np.array(info['object_positions'][object_of_interest][:3]).squeeze() # remove orientation of object

                # img = cv2.cvtColor(info["img"], cv2.COLOR_RGB2BGR)
                # img_name = "img_" + str(img_num) + ".jpg"
                # cv2.imwrite(os.path.join(imgfolder, img_name), img)
                if not np.round(t * 1/dt) % 500: # post every second
                    print("t: {}\nstate: {}\nobj_state: {}\naction: {}\n----".format(t, state, object_state, action))

                if np.linalg.norm(final_state-state) < 0.01:
                    json.dump(waypoints.tolist(), open(os.path.join(savefolder, "original_waypoints.json"), "w"))
                    json.dump(demo_traj, open(os.path.join(savefolder, "traj.json"), "w"))
                    json.dump(obj_traj, open(os.path.join(savefolder, "obj_traj.json"), "w"))

                    demo_traj = np.array(demo_traj)
                    obj_traj = np.array(obj_traj)
                    axs.plot(demo_traj[:, 1], demo_traj[:, 2], demo_traj[:, 3], 'b', label='Robot Traj')
                    axs.plot(obj_traj[:, 1], obj_traj[:, 2], obj_traj[:, 3], 'r', label=f'{object_of_interest} traj')
                    axs.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(savefolder, "result.jpg"))
                    break
                
                # save relevant info
                if not np.round(t * 1/dt) % 500:
                    demo_traj.append([t] + state.tolist())
                    obj_traj.append([t] + object_state.tolist())

                # step
                t += dt 
                img_num += 1
            sim.close()
            # # show resulting trajectories
            #     plt.show()