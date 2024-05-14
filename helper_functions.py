import numpy as np
import json
import os
import matplotlib.pyplot as plt
import gym, gym_panda
from utils import SQUISH_E, Trajectory, ManipulateReward
from explorers import ExplorationPolicyBO#, ExplorationPolicyCE
from test_ce_2 import ExplorationPolicyCE
from train_residual import distort_coords
# from test_cmaes import ExplorationPolicyCMAES
import re
import yaml

def solve_traj(cfg, args, prior_traj, object_of_interest, savefolder):

    # initialize rewards
    demofolder = os.path.join(cfg["save_data"]["DEMOS"], args.name)
    reward_fcn = ManipulateReward(demofolder)
        
    # load prior
    prior_trajectory = prior_traj.copy()
    contacts = prior_trajectory[:, -1]
    explore_inds = np.where(contacts==1)[0]
    explore_waypoints = prior_trajectory[explore_inds, 1:-1]

    num_points = len(explore_inds)
    env_size = cfg["env"]["SIZE"] * num_points
    reward_threshold = cfg["training"]["R_THRESH_BO"]
    
    obj_filename = os.path.join(demofolder, "obj_traj.json")
    with open(obj_filename, "r") as f:
        demo_obj_traj = np.array(json.load(f))
    demo_obj_traj = Trajectory(demo_obj_traj)

    all_limits = []
    for idx in range(num_points):
        t = prior_trajectory[explore_inds[idx], 0]
        object_position = demo_obj_traj.get_waypoint(t) + np.random.normal(0, 0.05, 3)
        limits = compute_limits(cfg, explore_waypoints[idx, :].squeeze(), object_position, explorer_type="BO")
        all_limits += limits
    explorer = ExplorationPolicyBO(env_size, all_limits, id=1, cfg=cfg, savefolder=savefolder)

    n_rollouts = cfg['training']['N_ROLLOUTS']
    rewards = -np.infty
    done = False
    roll_num = 0
    for roll_num in range(n_rollouts):
        rollout_dir = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder, str(roll_num))
        init_dirs([rollout_dir])

        if done:
            final_traj = goals.copy()
                        
            # return the prior, the posterior, the indices of points to explore,
            # rollout number at convergence, rewards at convergence
            return prior_trajectory, final_traj, explore_inds, \
                    roll_num, rewards, done
                
        # params to follow trajectory
        times = prior_trajectory[explore_inds, 0].squeeze()

        goals = prior_trajectory.copy()

        suggestion = explorer.ask()
        done = explorer.done
        
        suggestion = np.array(suggestion).reshape(num_points,3)
        print("Chosen suggestion: \n{}\n".format(suggestion))

        goals[explore_inds, 1:-1] = suggestion.copy()
        
        traj = goals.copy()
        rollout_traj, obj_traj = simulate(args, traj, object_of_interest)
        obj_traj = np.array(obj_traj)
        rewards = reward_fcn.compute_reward(obj_traj, times)
        explorer.tell(suggestion.flatten().tolist(), np.sum(rewards))
    
        print('rollout: {}, rewards: {}, threshold: {}'.format(roll_num, rewards, reward_threshold))
        
        results = dict()
        results["rewards"] = rewards
        results["suggestion"] = suggestion.tolist()
        results["roll"] = roll_num
        json.dump(results, open(os.path.join(rollout_dir, 'results.json'), 'w'))

    done = False
    suggestion = []
    rewards = []

    best_suggestion, best_reward = explorer.best_sample()
    best_suggestion = np.array(best_suggestion).reshape(num_points, 3)
    rewards.append(best_reward)

    best_traj = prior_trajectory.copy()
    best_traj[explore_inds, 1:-1] = best_suggestion.copy()        
    return prior_trajectory, best_traj, explore_inds, roll_num, rewards, done

def solve_manipulate(cfg, args, subtask, subtask_name, object_of_interest, savefolder, approach_traj, ablation=None):
    """
    cfg - config file (yaml) 
        Contains configuration parameters
    args - arguments (argparse) 
        See in main for required arguments
    subtask - Name of subtask to rollout (string) 
    """
    assert "manipulate" in subtask_name, ValueError("Need manipulate task, received {} instead.".format(subtask_name))

    # initialize rewards
    demofolder = os.path.join(cfg["save_data"]["DEMOS"], args.name)
    reward_fcn = ManipulateReward(demofolder)
        
    # load prior
    prior_trajectory = np.array(subtask["trajectory"])
    explore_waypoints = np.array(subtask["explore_waypoints"])
    explore_inds = np.array(subtask["explore_indices"])

    num_points = len(explore_inds)
    env_size = cfg["env"]["SIZE"]
    reward_threshold = cfg["training"]["R_THRESH_BO"]

    subtask_num = map(int, re.findall(r'\d+', subtask_name)[0])
    explorers = [None] * num_points
    obj_filename = os.path.join(demofolder, "obj_traj.json")
    with open(obj_filename, "r") as f:
        demo_obj_traj = np.array(json.load(f))
    demo_obj_traj = Trajectory(demo_obj_traj)
    for idx in range(num_points):
        t = prior_trajectory[explore_inds[idx], 0]
        object_position = demo_obj_traj.get_waypoint(t) + np.random.normal(0, 0.05, 3)
        # if ablation == "with_residual":
        #     hand_traj_dist = 0.05 
        #     object_dist = 0.05
        # else:
        hand_traj_dist = None
        object_dist = None
        limits = compute_limits(cfg, explore_waypoints[idx, :].squeeze(), object_position, explorer_type="BO", hand_traj_dist=hand_traj_dist, object_dist=object_dist)
        explorer = ExplorationPolicyBO(env_size, limits, id=idx, cfg=cfg, savefolder=savefolder)
        # explorer = ExplorationPolicyCMAES(explore_waypoints[idx, :].squeeze(), limits, n_suggestions=10,threshold=reward_threshold)
        explorers[idx] = explorer


    n_rollouts = cfg['training']['N_ROLLOUTS']
    rewards = -np.infty
    done = False
    roll_num = 0
    for roll_num in range(n_rollouts):
        rollout_dir = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder, subtask_name, str(roll_num))
        init_dirs([rollout_dir])

        if done:
            final_traj = goals.copy()
                        
            # return the prior, the posterior, the indices of points to explore,
            # rollout number at convergence, rewards at convergence
            return prior_trajectory, final_traj, explore_inds, \
                    roll_num, rewards, done
                
        # params to follow trajectory
        times = prior_trajectory[explore_inds, 0]

        goals = prior_trajectory.copy()
        suggestion = []
        dones = []
        for explorer in explorers:
            suggestion.append(explorer.ask())
            dones.append(explorer.done)
        
        suggestion = np.array(suggestion)
        goals[:, 1:-1] = suggestion.copy()
        
        print("Chosen suggestion: \n{}\n".format(suggestion))
        traj = np.row_stack((approach_traj, goals)) # add approach to pickup object
        rollout_traj, obj_traj = simulate(args, traj, object_of_interest)
        obj_traj = np.array(obj_traj)
        rewards = reward_fcn.compute_reward(obj_traj, times)
        for idx in range(num_points):
            explorers[idx].tell(suggestion[idx, :].tolist(), rewards[idx])
    
        print('rollout: {}, rewards: {}, threshold: {}'.format(roll_num, rewards, reward_threshold))
        
        results = dict()
        results["rewards"] = rewards
        results["suggestion"] = suggestion.tolist()
        results["roll"] = roll_num
        json.dump(results, open(os.path.join(rollout_dir, 'results.json'), 'w'))

        done = all(dones)
    done = False
    suggestion = []
    rewards = []
    for explorer in explorers:
        best_suggestion, best_reward = explorer.best_sample()
        suggestion.append(best_suggestion)
        rewards.append(best_reward)
    best_traj = prior_trajectory.copy()
    best_traj[explore_inds, 1:-1] = np.array(suggestion).copy()        
    return prior_trajectory, best_traj, explore_inds, roll_num, rewards, done

def solve_approach(cfg, args, subtask, subtask_name, object_of_interest, savefolder, explorer_type="CE", ablation=None):
    """
    cfg - config file (yaml) 
        Contains configuration parameters
    args - arguments (argparse) 
        See in main for required arguments
    subtask - Name of subtask to rollout (string) 
    """
    limit_x = cfg['env']['LIMIT_X']
    limit_y = cfg['env']['LIMIT_Y']
    limit_z = cfg['env']['LIMIT_Z']
    limits = [limit_x, limit_y, limit_z]

    # initialize rewards
    demofolder = os.path.join(cfg["save_data"]["DEMOS"], args.name)
        
    # load prior
    prior_trajectory = np.array(subtask["trajectory"])
    explore_waypoints = np.array(subtask["explore_waypoints"])
    explore_inds = np.array(subtask["explore_indices"])

    # object_location = distort_coords(args.object_positions[0,:3].copy(), limits)
    object_location = args.object_positions[0,:3] + np.random.normal(0, 0.05, args.object_positions[0,:3].shape)
    # if ablation == "with_residual":
    #     hand_traj_dist = 0.05 
    #     object_dist = 0.05
    # else:
    hand_traj_dist = None
    object_dist = None
    limits = compute_limits(cfg, explore_waypoints.squeeze(), object_location, explorer_type=explorer_type, hand_traj_dist=hand_traj_dist, object_dist=object_dist)
    env_size = cfg["env"]["SIZE"]

    subtask_num = map(int, re.findall(r'\d+', subtask_name)[0])
    if explorer_type == "CE":
        explorer = ExplorationPolicyCE(id=subtask_num, prior=explore_waypoints, limits=limits, env_size=env_size, savefolder=savefolder, cfg=cfg) 
    else:
        explorer = ExplorationPolicyBO(env_size, limits, id=subtask_num, cfg=cfg, savefolder=savefolder) 
    # explorer.plot_centroids(name=subtask_num, subtask="0_approach")

    n_rollouts = cfg['training']['N_ROLLOUTS']

    reward_threshold = cfg["training"]["R_THRESH_CE"]
    rewards = -np.infty
    for roll_num in range(n_rollouts):
        rollout_dir = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder, subtask_name, str(roll_num))
        init_dirs([rollout_dir])

        if rewards > reward_threshold:
            final_traj = goals.copy()
            done = True
            # return the prior, the posterior, the indices of points to explore,
            # rollout number at convergence, rewards at convergence
            return prior_trajectory, final_traj, explore_inds, \
                    roll_num, rewards, done
                
        # params to follow trajectory
        times = prior_trajectory[:, 0]
        
        goals = prior_trajectory.copy()
        suggestion = explorer.ask()
        goals[explore_inds, 1:-1] = suggestion.copy()
        
        rollout_traj, obj_traj = simulate(args, goals, object_of_interest)
        obj_traj = np.array(obj_traj)
        rollout_traj = np.array(rollout_traj)
        rewards = -np.linalg.norm(rollout_traj[-1, 1:-1] - obj_traj[-1, 1:])
        if explorer_type == "CE":
            explorer.tell(suggestion, rewards, obj_traj[:,1:])
        else:
            explorer.tell(suggestion.tolist(), rewards)
    
        print('rollout: {}, rewards: {}, threshold: {}'.format(roll_num, rewards, cfg['training']['R_THRESH_CE']))
        
        results = dict()
        results["rewards"] = rewards.tolist()
        results["suggestion"] = suggestion.tolist()
        results["roll"] = roll_num
        json.dump(results, open(os.path.join(rollout_dir, 'results.json'), 'w'))
    done = False
    if explorer_type == "CE":
        best_suggestion, best_reward, _ = explorer.best_sample()
    else:
        best_suggestion, best_reward = explorer.best_sample()
    best_traj = prior_trajectory.copy()
    best_traj[explore_inds, 1:-1] = best_suggestion.copy()
    return prior_trajectory, best_traj, explore_inds, roll_num, best_reward, done

def compute_limits(cfg, waypoints, object_location, explorer_type="CE", hand_traj_dist=None, object_dist=None):
    if explorer_type == "CE":
        if hand_traj_dist is None:
            LIMITS = cfg["explorer_CE"]["LIMITS"]
        if object_dist is None:
            object_dist = 0.07
    elif explorer_type =="BO":
        if hand_traj_dist is None:
            LIMITS = cfg["explorer_BO"]["LIMITS"]
        if object_dist is None:
            object_dist = 0.11
    if hand_traj_dist is not None:
        LIMITS = hand_traj_dist

    limit_x = cfg['env']['LIMIT_X']
    limit_y = cfg['env']['LIMIT_Y']
    limit_z = cfg['env']['LIMIT_Z']

    lower_limits = waypoints.copy()
    lower_limits -= LIMITS

    lower_limits[0] = np.clip(lower_limits[0], limit_x[0], limit_x[1])
    lower_limits[1] = np.clip(lower_limits[1], limit_y[0], limit_y[1])
    lower_limits[2] = np.clip(lower_limits[2], limit_z[0], limit_z[1])

    upper_limits = waypoints.copy()
    upper_limits += LIMITS

    upper_limits[0] = np.clip(upper_limits[0], limit_x[0], limit_x[1])
    upper_limits[1] = np.clip(upper_limits[1], limit_y[0], limit_y[1])
    upper_limits[2] = np.clip(upper_limits[2], limit_z[0], limit_z[1])

    object_lower_limits = object_location - object_dist
    object_upper_limits = object_location + object_dist
    
    bbox_lower_limits = np.minimum(object_lower_limits, waypoints)
    bbox_upper_limits = np.maximum(object_upper_limits, waypoints)

    bbox_lower_limits[0] = np.clip(bbox_lower_limits[0], lower_limits[0], upper_limits[0])
    bbox_lower_limits[1] = np.clip(bbox_lower_limits[1], lower_limits[1], upper_limits[1])
    
    if hand_traj_dist is None:
        bbox_lower_limits[2] = limit_z[0]
    else:
        bbox_lower_limits[2] = np.clip(bbox_lower_limits[2], lower_limits[2], upper_limits[2])

    bbox_upper_limits[0] = np.clip(bbox_upper_limits[0], lower_limits[0], upper_limits[0])
    bbox_upper_limits[1] = np.clip(bbox_upper_limits[1], lower_limits[1], upper_limits[1])
    bbox_upper_limits[2] = np.clip(bbox_upper_limits[2], lower_limits[2], upper_limits[2])
    # print("BBOX LOWER:\n{}\nBBOX_UPPER:\n{}\nwaypoints:\n{}\nOBJECT:\n{}\n".format(bbox_lower_limits, bbox_upper_limits, waypoints, object_location))
    # print("\n[*] Received prior\n {}".format(waypoints))
    # print("\n[*] Applying the following lower limits\n {}".format(lower_limits))
    # print("\n[*] Applying the following upper limits\n {}".format(upper_limits))

    limits = np.column_stack((bbox_lower_limits.flatten(), bbox_upper_limits.flatten()))
    limits = limits.tolist()
    return limits

'''    Simulate Trajectory    '''
def simulate(args, traj, object_of_interest, verbose=False):

    sim = gym.make('panda-v0', args=args)

    robot_state = sim.reset()
    initial_pos = robot_state[:3].copy()
    initial_ang = robot_state[3:7].copy()
    initial_grip = robot_state[-1].copy()

    times = traj[:,0]
    traj_fn = Trajectory((traj))

    t = 0
    dt = args.dt # DO NOT CHANGE
    max_t = times[-1] + 2.0 # give two extra seconds for sim 
    final_state = traj[-1,1:] # remove time for state

    robot_traj = []
    obj_traj = []
    while True:
        target = traj_fn.get_waypoint(t) # contains [x, y, z, gripper]
        linear = target[:3]
        angular = initial_ang.copy() # no orientation in traj fn
        grip = target[-1]
        action = np.hstack((linear, angular, [grip]))
        
        robot_state, _,_, info = sim.step(action)
        while not grip == robot_state[-1]: # wait for gripper to open/close
            robot_state, _,_, info = sim.step(action)
        state = np.hstack((robot_state[:3], robot_state[-1]))
        object_state = np.array(info['object_positions'][object_of_interest][:3]) # remove orientation of object

        if verbose:
            if not np.round(t * 1/dt) % 1000: # post every second
                print("t: {}\nstate: {}\nobj_state: {}\naction: {}\n----".format(t, state, object_state, action))

        if np.linalg.norm(final_state-state) < 0.015 or t > max_t:
            sim.close()
            return robot_traj, obj_traj
        
        # save relevant info
        robot_traj.append([t] + state.tolist())
        obj_traj.append([t] + object_state.tolist())

        # step
        t += dt 

''' Numerically sort strings'''
def num_sort(input_string):
    return list(map(int, re.findall(r'\d+', input_string)))[0]

'''    Load demo    '''
def load_prior(args, cfg):

    filename = os.path.join(cfg["save_data"]["DEMOS"], args.name, "traj.json")
    with open(filename, 'r') as f:
        clean_traj = np.array(json.load(f))
    return clean_traj

'''    Limit traj to robot workspace    '''
def constrain_traj(traj):
    traj = np.array(traj)
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    limit_x = cfg['env']['LIMIT_X']
    limit_y = cfg['env']['LIMIT_Y']
    limit_z = cfg['env']['LIMIT_Z']

    traj[:, 1] = np.clip(traj[:, 1], limit_x[0], limit_x[1])
    traj[:, 2] = np.clip(traj[:, 2], limit_y[0], limit_y[1])
    traj[:, 3] = np.clip(traj[:, 3], limit_z[0], limit_z[1])
    return traj

'''    Add noise to clean prior    '''
def distort_traj(traj, indices, mean=0., var=0.01):
    traj = np.array(traj)
    distorted_traj = traj.copy()
    indices = np.array(indices)
    for idx in indices:
        noise = []
        for _ in range(3):
            mean = np.random.uniform(-0.03, 0.03) # atleast be 3cm off
            noise.append(np.random.normal(mean, var))
        distorted_traj[idx, 1:4] += np.array(noise)#np.random.normal(mean, var, distorted_traj[idx, 1:4].shape)
    distorted_traj = constrain_traj(distorted_traj)
    return distorted_traj

'''    Break into subtasks    '''
def generate_subtasks(traj):

    # find all points where contact = 1
    contacts = traj[:,-1]
    explore_inds = np.where(contacts==1)[0]

    subtasks = dict()

    # split into approach and manipulate
    for subtask_idx in range(2):
        if subtask_idx == 0:
            subtask_name = "0_approach"
            explore_ind = [explore_inds[0]]  
            start = 0
            end = explore_inds[1]
            explore_waypoints = traj[None, explore_inds[0], 1:-1]
            subtask_traj = traj[start:end+1, :].copy()
            subtask_traj[-1,3] = 1.17 # pick a point high in the air
        else:
            subtask_name = "0_manipulate"
            explore_ind = np.arange(len(explore_inds[1:])).tolist()
            start = explore_inds[1]
            end = explore_inds[-1]
            if start == end:
                explore_waypoints = traj[None, explore_inds[1], 1:-1]
                subtask_traj = traj[None, start, :].copy()
            else:
                explore_waypoints = traj[explore_inds[1:], 1:-1]
                subtask_traj = traj[start:end+1, :].copy()

        
        subtask_info = dict()
        subtask_info["trajectory"] = subtask_traj.tolist()
        subtask_info["explore_waypoints"] = explore_waypoints.tolist()
        subtask_info["explore_indices"] = explore_ind
        subtasks[subtask_name] = subtask_info

    return subtasks

'''    Identify the points of interest    '''
def extract_poi(subtask_traj, subtask_name):
    explore_ind = np.where(subtask_traj[:, -1] == 1)[0].tolist()
    if 'approach' in subtask_name:
        explore_ind = [explore_ind[0]]
        explore_waypoints = subtask_traj[explore_ind, 1:-1]
    elif 'manipulate' in subtask_name:
        explore_waypoints = subtask_traj[explore_ind[1:], 1:-1]
        subtask_traj = subtask_traj[explore_ind[1:], :] # remove pickup point from prior
        explore_ind = [item-1 for item in explore_ind[1:]] # update to remove first point
 
    if len(explore_waypoints.shape) == 1:
        explore_waypoints = explore_waypoints[None, :]
    
    return subtask_traj, explore_waypoints, explore_ind

'''    Make the required directories    '''
def init_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

'''    Saving numpy array in json    '''
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)