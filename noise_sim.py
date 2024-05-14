import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from helper_functions import load_prior, generate_subtasks, solve_approach, solve_manipulate, distort_traj, init_dirs, NumpyArrayEncoder
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

def wayvil(args, cfg, prior_traj, object_of_interest, savefolder, skip_tasks=[]):

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
        print("Prior: \n{} \nSubtask: {} \nDistorted Subtask Traj: \n{} \nIndices: {}".format(prior_traj, sb, np.array(traj), indices))
        if "approach" in sb:
            prior_trajectory, final_traj, explore_inds, roll_num, rewards, done = solve_approach(cfg, args, subtasks[sb], sb, object_of_interest, savefolder)
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
    noise_range = [0.05, 0.07, 0.1, 0.15, 0.2]

    lamda = args.points / traj.shape[0]
    compressor = SQUISH_E()
    traj = compressor.squish(traj, lamda=lamda)
    print("Compressed Traj:\n {}".format(traj))
    timestr = time.strftime('_%Y%m%d_%H%M%S')
    
    final_folder = os.path.join(cfg['save_data']['RESULTS'])
    init_dirs([final_folder])
    
    results = dict()
    for noise_var in noise_range:
        results_savename = os.path.join(final_folder, "noise_{}.json".format(args.name))
        if os.path.isfile(results_savename):
            results = json.load(open(results_savename, "r"))
        
            if str(noise_var) in results.keys():
                all_dones = results[str(noise_var)]["done"]
                all_rollouts = results[str(noise_var)]["rollouts"]
            else:
                all_dones = []
                all_rollouts = []
            if len(all_rollouts) >= args.trials:
                print("Trials for {} completed previously, skipping.".format(noise_var))
                continue
            else:
                start = len(all_rollouts)
                print("Running {} from {} trials.".format(noise_var, start))
        else:
            all_dones = []
            all_rollouts = []
            start = 0
        args.noise = noise_var
        for trial in range(start, args.trials):
            savefolder = os.path.basename(__file__).replace(".py", "") + timestr + "_" + str(noise_var) + "_" + str(trial)
            # save config used
            savepath = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder)
            init_dirs([savepath])
            yaml.safe_dump(cfg, open(os.path.join(savepath, "cfg.yaml"), "w"))
            json.dump(vars(args), open(os.path.join(savepath, "args.json"), "w"), cls=NumpyArrayEncoder, indent=2)

            res = wayvil(args, cfg, traj, object_of_interest, savefolder)
            savename = os.path.join(savepath, "results.json")
            json.dump(res, open(savename, "w"))
            dones = all([val for key, val in res["dones"].items()])
            rollouts = sum([rollout for key, rollout in res["rollouts"].items()])
            print("noise: {}, trial: {}, success: {}, rollouts: {}".format(noise_var, trial, dones, rollouts))
            all_dones.append(dones)
            all_rollouts.append(rollouts)

            noise_results = dict()
            noise_results["done"] = all_dones
            noise_results["rollouts"] = all_rollouts
            results[str(noise_var)] = noise_results

            json.dump(results, open(results_savename, "w"))
            