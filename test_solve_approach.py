import numpy as np
import argparse
import os
import time
import yaml, json
from helper_functions import solve_approach, generate_subtasks, load_prior, init_dirs, NumpyArrayEncoder, distort_traj
import warnings
warnings.simplefilter("ignore")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--name", required=True, help="Name of task")
    args = parser.parse_args()
    return args

args = parse_args()
cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

# Add objects to sim
args.objects = ["025_mug"]
args.object_positions = np.array([[0.4, -0.3, 0.65, 0.0, 0.0, 1.0, 0.0]])

object_of_interest = args.objects[0]
compressed_traj, clean_traj = load_prior(args, cfg)
subtasks = generate_subtasks(compressed_traj)

print(subtasks)

subtask_name = "0_approach"
subtask = subtasks[subtask_name]
subtask["trajectory"] = distort_traj(subtask["trajectory"], indices=subtask["explore_indices"], mean=0.0, var=0.05)

timestr = time.strftime('_%Y%m%d_%H%M%S')
savefolder = os.path.basename(__file__).replace(".py", "") + timestr

    
# save config used
savepath = os.path.join(cfg['save_data']['ROLLOUTS'], savefolder)
init_dirs([savepath])
yaml.safe_dump(cfg, open(os.path.join(savepath, "cfg.yaml"), "w"))
json.dump(vars(args), open(os.path.join(savepath, "args.json"), "w"), cls=NumpyArrayEncoder, indent=2)

solve_approach(cfg, args, subtask, subtask_name, object_of_interest, savefolder)
