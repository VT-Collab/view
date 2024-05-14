import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from helper_functions import load_prior, generate_subtasks, solve_approach, solve_manipulate, solve_traj, distort_traj, init_dirs, NumpyArrayEncoder
from utils import SQUISH_E
import yaml
import time
import warnings
warnings.simplefilter("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--name", required=True, help="Name of demonstration")
    parser.add_argument("--points", required=True, type=int, help="Number of points to keep after compressing")
    parser.add_argument("--dt", default=1e-4, type=float, help="Number of points to keep after compressing")
    parser.add_argument("--trials", type=int, required=True, help="Number of trials for each noise value")
    args = parser.parse_args()
    return args

def no_waypoint_split(args, cfg, prior_traj, object_of_interest, savefolder):
    
    final_trajectories = dict()
    final_rollouts = dict()
    final_rewards = dict()
    final_dones = dict()
    final_n_points = dict()

    prior_trajectory, final_traj, explore_inds,\
          roll_num, rewards, done = solve_traj(cfg, args, prior_traj, object_of_interest, savefolder)
    final_trajectories["task"] = final_traj.tolist()
    final_rollouts["task"] = roll_num
    final_rewards["task"] = rewards
    final_dones["task"] = done
    final_n_points["task"] = len(explore_inds)

    result = dict()
    result["traj"] = final_trajectories
    result["rollouts"] = final_rollouts
    result["rewards"] = final_rewards
    result["dones"] = final_dones
    result["n_points"] = final_n_points

    return result

def reward_only(args, cfg, prior_traj, object_of_interest, savefolder, skip_tasks=[]):
    
    contact_indices = np.where(prior_traj[:,-1]==1)[0].tolist()
    distorted_traj = distort_traj(prior_traj, contact_indices, var=args.noise)
    subtasks = generate_subtasks(distorted_traj)

    final_trajectories = dict()
    final_rollouts = dict()
    final_rewards = dict()
    final_dones = dict()
    final_n_points = dict()
    for sb in sorted(subtasks.keys()):
        subtask_type = sb.split("_")[1]
        if subtask_type in skip_tasks:
            continue
        traj = subtasks[sb]["trajectory"]
        indices = subtasks[sb]["explore_indices"]
        # subtasks[sb]["trajectory"] = distorted_traj.tolist()
        print("Subtask: {}\nPrior:\n{}\nDistorted_prior:\n{}\nIndices:{}".format(sb, np.array(traj), np.array(distorted_traj), indices))
        if "approach" in sb:
            prior_trajectory, final_traj, explore_inds, roll_num, rewards, done = solve_approach(cfg, args, subtasks[sb], sb, object_of_interest, savefolder, explorer_type="BO")
        if "manipulate" in sb:
            if "approach" in skip_tasks:
                approach_traj = np.array(subtasks[sb.replace("manipulate", "approach")]["trajectory"])[:-1,:]

            else:
                approach_traj = np.array(final_trajectories[sb.replace("manipulate", "approach")])[:-1,:] # ignore last point
            prior_trajectory, final_traj, explore_inds, roll_num, rewards, done = solve_manipulate(cfg, args, subtasks[sb], sb, object_of_interest, \
                                                                                             savefolder, approach_traj)
        final_trajectories[sb] = final_traj.tolist()
        final_rollouts[sb] = roll_num
        final_rewards[sb] = rewards
        final_dones[sb] = done
        final_n_points[sb] = len(indices)
        
        if not done:
            break
    
    result = dict()
    result["traj"] = final_trajectories
    result["rollouts"] = final_rollouts
    result["rewards"] = final_rewards
    result["dones"] = final_dones
    result["n_points"] = final_n_points

    return result

if __name__ =="__main__":
    args = parse_args()
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

    
    # Add objects to sim
    args.objects = ["025_mug"]
    args.object_positions = np.array([[0.4, -0.3, 0.68, 0.0, 0.0, 1.0, 0.0]])
    
    object_of_interest = args.objects[0]
    
    traj = load_prior(args, cfg)

    lamda = args.points / traj.shape[0]
    compressor = SQUISH_E()
    traj = compressor.squish(traj, lamda=lamda)
    print("Compressed Traj:\n {}".format(traj))
    timestr = time.strftime('_%Y%m%d_%H%M%S')
    results = dict()
    args.noise = 0.2

    final_folder = os.path.join(cfg['save_data']['RESULTS'])
    init_dirs([final_folder])

    exploration_types = ["reward_only", "no_waypoint_split"]

    for exploration_type in exploration_types:
        results_savename = os.path.join(final_folder, "exploration_{}.json".format(args.name))
        if os.path.isfile(results_savename):
            results = json.load(open(results_savename, "r"))
        
            if str(exploration_type) in results.keys():
                all_dones = results[str(exploration_type)]["done"]
                all_rollouts = results[str(exploration_type)]["rollouts"]
            else:
                all_dones = []
                all_rollouts = []
            if len(all_rollouts) >= args.trials:
                print("Trials for {} completed previously, skipping.".format(exploration_type))
                continue
            else:
                start = len(all_rollouts)
                print("Running {} from {} trials.".format(exploration_type, start))
        else:
            all_dones = []
            all_rollouts = []
            start = 0

        for trial in range(start, args.trials):
            savefolder = os.path.basename(__file__).replace(".py", "") + timestr + "_" + exploration_type + "_" + str(trial)
            # save config used
            savepath = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder)
            init_dirs([savepath])
            yaml.safe_dump(cfg, open(os.path.join(savepath, "cfg.yaml"), "w"))
            json.dump(vars(args), open(os.path.join(savepath, "args.json"), "w"), cls=NumpyArrayEncoder, indent=2)
            
            if exploration_type == "reward_only":
                res = reward_only(args, cfg, traj, object_of_interest, savefolder)
            elif exploration_type == "no_waypoint_split":
                res = no_waypoint_split(args, cfg, traj, object_of_interest, savefolder)
            savename = os.path.join(savepath, "results.json")
            json.dump(res, open(savename, "w"))
            dones = all([val for key, val in res["dones"].items()])
            rollouts = sum([rollout for key, rollout in res["rollouts"].items()])
            print("exploration_type: {}, trial: {}, success: {}, rollouts: {}".format(exploration_type, trial, dones, rollouts))
            all_dones.append(dones)
            all_rollouts.append(rollouts)

            exploration_results = dict()
            exploration_results["done"] = all_dones
            exploration_results["rollouts"] = all_rollouts
            results[str(exploration_type)] = exploration_results

            json.dump(results, open(results_savename, "w"))