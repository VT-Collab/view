import yaml
import os
import numpy as np
import argparse
from helper_functions import load_prior, generate_subtasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of demonstration")
    args = parser.parse_args()

    return args

args = parse_args()
cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

distorted_traj, compressed_traj, clean_traj = load_prior(args, cfg, noise_mean=0.02, noise_var=0.01)
prior_traj = distorted_traj.copy()

subtasks = generate_subtasks(prior_traj)
print(subtasks)
