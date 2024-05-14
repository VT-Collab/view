import matplotlib.pyplot as plt
import numpy as np
import json
import os
from helper_functions import num_sort

parent_dir = "./data/results"
all_results = os.listdir(parent_dir)

# residual_files = [item for item in all_results if "residual" in item]
# noise_files = [item for item in all_results if "noise" in item]

ablation = "residual"
traj_types = ["pick", "push", "move"]
filenames = [ablation + "_" + traj_type + ".json" for traj_type in traj_types]

task_names = []
success = dict()
rollouts = dict()
variances = dict()

for filename in filenames:
    result = json.load(open(os.path.join(parent_dir, filename)))
    
    task_name = os.path.splitext(filename)[0].split("_")[1] # get the task name
    task_names.append(task_name)
    
    for key in result:
        if key not in success:
            success[key] = []
        success[key].append(np.count_nonzero(result[key]["done"]) * 0.02)
        if key not in rollouts:
            rollouts[key] = []
        successful_rollouts = np.array(result[key]["rollouts"])[np.nonzero(result[key]["done"])] - 1
        rollouts[key].append(np.mean(successful_rollouts))
        if key not in variances:
            variances[key] = []
        sd = np.std(successful_rollouts)
        variances[key].append(sd / np.sqrt(len(successful_rollouts)))

print(success)
print(rollouts)

# noise_levels = sorted(rollouts.keys(), key=num_sort)
noise_levels = sorted(rollouts.keys())


width = 0.15
multiplier = 0
x = np.arange(3)
fig, axs = plt.subplots(1, 2)
for key in noise_levels:
    offset = width * multiplier
    rects = axs[0].bar(x+offset, rollouts[key], width, label=key, yerr=variances[key]) # for rollouts
    # axs[0].bar_label(rects, padding=3)

    rects = axs[1].bar(x+offset, success[key], width, label=key) # for success rate
    # axs[1].bar_label(rects, padding=3)
    multiplier += 1
    
axs[0].set_ylabel('Rollouts')
axs[0].set_title('Rollouts')

axs[1].set_ylabel('Success Rate')
axs[1].set_title('Success rate')

axs[0].set_xticks(x + width * (len(noise_levels)-1)/2, task_names)
axs[1].set_xticks(x + width * (len(noise_levels)-1)/2, task_names)

axs[0].legend()

plt.show()

# task_names = []
# noresidual_rollouts = []
# residual_rollouts = []
# noresidual_success = []
# residual_success = []



# width = 0.25
# multiplier = 0

# x = np.arange(3)


# for residual_file in residual_files:
#     result = json.load(open(os.path.join(parent_dir, residual_file)))

#     with_residual = result["with_residual"]
#     without_residual = result["without_residual"]

#     residual_success.append(np.count_nonzero(with_residual["done"]) * 0.02)
#     noresidual_success.append(np.count_nonzero(without_residual["done"]) * 0.02)

#     # get only sucessful rollouts
#     n_rollouts_residual = np.array(with_residual["rollouts"])[np.nonzero(with_residual["done"])] - 1
#     n_rollouts_noresidual = np.array(without_residual["rollouts"])[np.nonzero(without_residual["done"])] - 1
    
#     residual_rollouts.append(np.mean(n_rollouts_residual))
#     noresidual_rollouts.append(np.mean(n_rollouts_noresidual))
#     task_name = os.path.splitext(residual_file)[0].split("_")[1] # get the task name
#     task_names.append(task_name)
#     print(task_name, np.mean(n_rollouts_residual), np.mean(n_rollouts_noresidual))


# # plotting
# offset = width * multiplier
# rects = axs[0].bar(x+offset, noresidual_rollouts, width, label='no residual') # for rollouts
# axs[0].bar_label(rects, padding=3)

# rects = axs[1].bar(x+offset, noresidual_success, width, label='no residual') # for success rate
# axs[1].bar_label(rects, padding=3)

# multiplier += 1
# offset = width * multiplier
# rects = axs[0].bar(x+offset, residual_rollouts, width, label='residual')
# axs[0].bar_label(rects, padding=3)

# rects = axs[1].bar(x+offset, residual_success, width, label='residual') # for success rate
# axs[1].bar_label(rects, padding=3)
