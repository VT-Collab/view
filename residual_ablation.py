import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
from nn_utils import ResidualNet
from train_residual import distort_coords
from helper_functions import load_prior, num_sort, generate_subtasks, solve_approach, solve_manipulate, init_dirs, NumpyArrayEncoder
from utils import SQUISH_E
import yaml
import time
import warnings
warnings.simplefilter("ignore")

device = "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--dt", default=1e-4, type=float, help="Number of points to keep after compressing")
    parser.add_argument("--model-name", default="model.pkl", type=str, help="Name of the residual model")
    args = parser.parse_args()
    return args

def wayvil(args, cfg, prior_traj, object_of_interest, savefolder, ablation, skip_tasks=[]):

    subtasks = generate_subtasks(prior_traj)

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
            prior_trajectory, final_traj, explore_inds, roll_num, rewards, done = solve_approach(cfg, args, subtasks[sb], sb, object_of_interest, savefolder, ablation=ablation)
        if "manipulate" in sb:
            if "approach" in skip_tasks:
                approach_traj = np.array(subtasks[sb.replace("manipulate", "approach")]["trajectory"])[:-1,:]

            else:
                approach_traj = np.array(final_trajectories[sb.replace("manipulate", "approach")])[:-1,:] # ignore last point
            prior_trajectory, final_traj, explore_inds, roll_num, rewards, done = solve_manipulate(cfg, args, subtasks[sb], sb, object_of_interest, \
                                                                                             savefolder, approach_traj, ablation=ablation)
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
    timestr = time.strftime('_%Y%m%d_%H%M%S')

    args = parse_args()
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

    final_folder = os.path.join(cfg['save_data']['RESULTS'])
    init_dirs([final_folder])

    limit_x = cfg['env']['LIMIT_X']
    limit_y = cfg['env']['LIMIT_Y']
    limit_z = cfg['env']['LIMIT_Z']
    
    lower_limits = [limit_x[0], limit_y[0], limit_z[0]]
    upper_limits = [limit_x[1], limit_y[1], limit_z[1]]

    ablation_types = ["with_residual", "without_residual"]

    # setup residual network
    savename = os.path.join("./models", args.model_name)
    model = ResidualNet().to(device)
    model_dict = torch.load(savename, map_location=device)
    model.load_state_dict(model_dict)
    model.eval()

    # get all demo folders
    all_demo_folders = os.listdir("./data/demos")
    traj_types = ["move", "push", "pick"]
    for traj_type in traj_types:
        demo_folders = sorted([item for item in all_demo_folders if traj_type in item], key=num_sort)
        results = dict()

        for ablation in ablation_types:
            results_savename = os.path.join(final_folder, "residual_{}.json".format(traj_type))
            if os.path.isfile(results_savename):
                results = json.load(open(results_savename, "r"))
            
                if str(ablation) in results.keys():
                    all_dones = results[str(ablation)]["done"]
                    all_rollouts = results[str(ablation)]["rollouts"]
                else:
                    all_dones = []
                    all_rollouts = []
                if len(all_rollouts) >= len(demo_folders):
                    print("Trials for {} completed previously, skipping.".format(ablation))
                    continue
                else:
                    start = len(all_rollouts)
                    print("Running {} from {} trials.".format(ablation, start))
            else:
                all_dones = []
                all_rollouts = []
                start = 0

            
            for folder_idx in range(start, len(demo_folders)):
                savefolder = os.path.basename(__file__).replace(".py", "") + timestr + "_" + str(ablation) + "_" + str(folder_idx)
                # save config used
                savepath = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder)
                init_dirs([savepath])
                yaml.safe_dump(cfg, open(os.path.join(savepath, "cfg.yaml"), "w"))
                json.dump(vars(args), open(os.path.join(savepath, "args.json"), "w"), cls=NumpyArrayEncoder, indent=2)
                
                demo_folder = demo_folders[folder_idx]
                args.name = demo_folder
                if "pick" in args.name:
                    args.points = 3
                elif "push" in args.name:
                    args.points = 4
                elif "move" in args.name:
                    args.points = 5

                traj = load_prior(args, cfg)
                lamda = args.points / traj.shape[0]
                compressor = SQUISH_E()
                traj = compressor.squish(traj, lamda=lamda)
                print("Compressed Traj:\n {}".format(traj))

                # distord coords
                distorted_traj = traj.copy()
                limits = [limit_x, limit_y, limit_z]
                # get contact_idxs
                contact_idxs = np.where(distorted_traj[:,-1] == 1)[0].tolist()
                for contact_idx in contact_idxs:
                    distorted_traj[contact_idx, 1:-1] = distort_coords(distorted_traj[contact_idx, 1:-1], limits)

                print("Distorted Traj:\n{}".format(distorted_traj))

                if ablation == "with_residual":
                    distorted_waypoints_tensor = torch.FloatTensor(distorted_traj[contact_idxs, 1:-1])
                    corrected_waypoints = model.get_residual(distorted_waypoints_tensor).detach().numpy()
                    corrected_traj = distorted_traj.copy()
                    corrected_traj[contact_idxs, 1:-1] = corrected_waypoints
                    traj_to_send = corrected_traj
                    print("Corrected Traj:\n {}".format(traj_to_send))
                else:
                    traj_to_send = distorted_traj.copy()

                obj_filename = os.path.join(cfg["save_data"]["DEMOS"], demo_folder, "obj_traj.json")
                with open(obj_filename, "r") as f:
                    demo_obj_traj = json.load(f)
                
                object_position = demo_obj_traj[0][1:]
                # Add objects to sim
                args.objects = ["025_mug"]
                args.object_positions = np.array([object_position + [0.0, 0.0, 1.0, 0.0]])
                object_of_interest = args.objects[0]
            
                res = wayvil(args, cfg, traj_to_send, object_of_interest, savefolder, ablation)
                savename = os.path.join(savepath, "results.json")
                json.dump(res, open(savename, "w"))
                dones = all([val for key, val in res["dones"].items()])
                rollouts = sum([rollout for key, rollout in res["rollouts"].items()])
                print("ablation: {}, demo_name: {}, success: {}, rollouts: {}".format(ablation, args.name, dones, rollouts))
                all_dones.append(dones)
                all_rollouts.append(rollouts)

                noise_results = dict()
                noise_results["done"] = all_dones
                noise_results["rollouts"] = all_rollouts
                results[str(ablation)] = noise_results

                json.dump(results, open(results_savename, "w"))
                