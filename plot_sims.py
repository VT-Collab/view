import json
import numpy as np
import matplotlib.pyplot as plt


noise_pick_data = json.load(open("./data/results/results_final/noise_pick.json"))
noise_move_data = json.load(open("./data/results/results_final/noise_move.json"))

tasks = ["Pick, Move"]

fig, axs = plt.subplots(1,2)

noise_levels = []
success_rate = []
for key in noise_pick_data.keys():
    done_count = sum(noise_pick_data[key]["done"][:100])
    rollouts = noise_pick_data[key]["rollouts"][:100]
    total_count = len(noise_pick_data[key]["done"][:100])
    noise_levels.append(key)
    success_rate.append(done_count)
    print("{}: done - {}, total- {}, n_rollouts- {}".format(key, done_count, total_count, np.mean(rollouts)))

x = np.arange(len(noise_levels))
width = 0.25

axs[0].bar(x, success_rate, width)
axs[0].set_title("Pick")

success_rate = []
for key in noise_move_data.keys():
    done_count = sum(noise_move_data[key]["done"][:100])
    rollouts = noise_move_data[key]["rollouts"][:100]
    total_count = len(noise_move_data[key]["done"][:100])
    success_rate.append(done_count)
    print("{}: done - {}, total- {}, n_rollouts- {}".format(key, done_count, total_count, np.mean(rollouts)))

axs[1].bar(x, success_rate, width)
axs[1].set_title("Move")

for ax in axs:
    ax.set_xlabel("Noise Levels")
    ax.set_ylabel("Success Rate")
    ax.set_xticks(x, noise_levels)


squishe_move_data = json.load(open("./data/results/no_squishe_move.json"))


fig, axs = plt.subplots()

method = []
success_rate = []
for key in squishe_move_data.keys():
    done_count = sum(squishe_move_data[key]["done"][:100])
    rollouts = squishe_move_data[key]["rollouts"][:100]
    total_count = len(squishe_move_data[key]["done"][:100])
    method.append("Sampling rate {}".format(key))
    success_rate.append(done_count)
    print("{}: done - {}, total- {}, n_rollouts- {}".format(key, done_count, total_count, np.mean(rollouts)))

method.append("Ours")
success_rate.append(sum(noise_move_data["0.15"]["done"][:100]))

x = np.arange(len(method))
axs.bar(x, success_rate, width)
axs.set_title("Move")

axs.set_xlabel("Methods")
axs.set_ylabel("Success Rate")
axs.set_xticks(x, method)

x = np.arange(len(noise_levels))

exploration_pick_data = json.load(open("./data/results/exploration_pick.json"))
exploration_move_data = json.load(open("./data/results/results_final/exploration_move.json"))


fig, axs = plt.subplots(1,2)

exploration_types = []
success_rate = []
for key in exploration_pick_data.keys():
    done_count = sum(exploration_pick_data[key]["done"][:100])
    rollouts = exploration_pick_data[key]["rollouts"][:100]
    total_count = len(exploration_pick_data[key]["done"][:100])
    exploration_types.append(key)
    success_rate.append(done_count)
    print("{}: done - {}, total- {}, n_rollouts- {}".format(key, done_count, total_count, np.mean(rollouts)))

exploration_types.append("Ours")
success_rate.append(sum(noise_pick_data["0.15"]["done"][:100]))

x = np.arange(len(exploration_types))

axs[0].bar(x, success_rate, width)
axs[0].set_title("Pick")

success_rate = []

for key in exploration_move_data.keys():
    done_count = sum(exploration_move_data[key]["done"][:100])
    rollouts = exploration_move_data[key]["rollouts"][:100]
    total_count = len(exploration_move_data[key]["done"][:100])
    success_rate.append(done_count)
    print("{}: done - {}, total- {}, n_rollouts- {}".format(key, done_count, total_count, np.mean(rollouts)))

success_rate.append(sum(noise_move_data["0.15"]["done"][:100]))

axs[1].bar(x, success_rate, width)
axs[1].set_title("Move")

for ax in axs:
    ax.set_xlabel("Exploration type")
    ax.set_ylabel("Success Rate")
    ax.set_xticks(x, exploration_types)

# plt.show()