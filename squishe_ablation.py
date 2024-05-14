import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from helper_functions import load_prior, init_dirs, NumpyArrayEncoder
from wayvil import wayvil
from utils import SQUISH_E
import yaml
import time
import warnings
warnings.simplefilter("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--name", required=True, help="Name of demonstration")
    parser.add_argument("--dt", default=1e-4, type=float, help="Number of points to keep after compressing")
    parser.add_argument("--trials", type=int, required=True, help="Number of trials for each noise value")
    parser.add_argument("--noise", default=0.15, type=float, help="Noise in trajectory")
    args = parser.parse_args()

    return args

def sample(time_gap, traj):
    contacts = traj[:, -1]
    contact_switch_idxs = np.where(contacts[1:] != contacts[:-1])[0] + 1
    compressed_traj = []
    compressed_traj.append(traj[0, :].tolist())
    compressed_traj.append(traj[contact_switch_idxs[0], :].tolist())
    start_idx = contact_switch_idxs[0]
    end_idx = traj.shape[0]
    start_time = traj[start_idx, 0].copy()
    for idx in range(start_idx, end_idx):
        curr_time = traj[idx, 0].copy()
        if curr_time - start_time >= time_gap:
            compressed_traj.append(traj[idx, :].tolist())
            start_time = traj[idx, 0].copy()
    return np.array(compressed_traj)

if __name__ =="__main__":
    args = parse_args()
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

    sample_times = [0.2, 0.1, 0.05]

    # Add objects to sim
    args.objects = ["025_mug"]
    args.object_positions = np.array([[0.4, -0.3, 0.68, 0.0, 0.0, 1.0, 0.0]])
    
    object_of_interest = args.objects[0]
    
    final_folder = os.path.join(cfg['save_data']['RESULTS'])
    init_dirs([final_folder])

    traj = load_prior(args, cfg)
    timestr = time.strftime('_%Y%m%d_%H%M%S')
    results = dict()
    for sample_time in sample_times:
        sampled_traj = sample(sample_time, traj)
        results_savename = os.path.join(final_folder, "no_squishe_{}.json".format(args.name))
        if os.path.isfile(results_savename):
            results = json.load(open(results_savename, "r"))
        
            if str(sample_time) in results.keys():
                all_dones = results[str(sample_time)]["done"]
                all_rollouts = results[str(sample_time)]["rollouts"]
            else:
                all_dones = []
                all_rollouts = []
            if len(all_rollouts) >= args.trials:
                print("Trials for {} completed previously, skipping.".format(sample_time))
                continue
            else:
                start = len(all_rollouts)
                print("Running {} from {} trials.".format(sample_time, start))
        else:
            all_dones = []
            all_rollouts = []
            start = 0
        n_points = 0
        for trial in range(start, args.trials):
            savefolder = os.path.basename(__file__).replace(".py", "") + timestr + "_sample_time_" + str(sample_time) + "_" + str(trial)
            # save config used
            savepath = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder)
            init_dirs([savepath])
            yaml.safe_dump(cfg, open(os.path.join(savepath, "cfg.yaml"), "w"))
            json.dump(vars(args), open(os.path.join(savepath, "args.json"), "w"), cls=NumpyArrayEncoder, indent=2)

            res = wayvil(args, cfg, sampled_traj, object_of_interest, savefolder)
            savename = os.path.join(savepath, "results.json")
            json.dump(res, open(savename, "w"))
            dones = all([val for key, val in res["dones"].items()])
            rollouts = sum([rollout for key, rollout in res["rollouts"].items()])
            print("sample time: {}, trial: {}, success: {}, rollouts: {}".format(sample_time, trial, dones, rollouts))
            all_dones.append(dones)
            all_rollouts.append(rollouts)
            n_points = res["n_points"]

            sample_results = dict()
            sample_results["done"] = all_dones
            sample_results["rollouts"] = all_rollouts
            sample_results["n_points"] = n_points
            results[str(sample_time)] = sample_results

            json.dump(results, open(results_savename, "w"))