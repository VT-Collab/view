import numpy as np
import matplotlib.pyplot as plt


# # single object data
# tasks = ["pick", "push", "move"]
# prior_grasping = np.array([0., 0., 0.]) + 0.1
# whirl_grasping = np.array([5./9., 3./9., 6./9.]) + 0.1
# ours_bo_grasping = np.array([0, 4./9., 5./9.]) + 0.1
# ours_grasping = np.array([1., 8./9., 1.]) + 0.1

# prior_task_success = np.array([0., 0., 0.]) + 0.1
# whirl_task_success = np.array([4./9., 0., 0.]) + 0.1
# ours_bo_task_success = np.array([0, 4./9., 4./9.]) + 0.1
# ours_task_success = np.array([1., 7./9., 7./9.]) + 0.1

# # multi object data
# tasks = ["Multi Object"]
# prior_grasping = np.array([0.]) + 0.1
# whirl_grasping = np.array([4./12.]) + 0.1
# ours_bo_grasping = np.array([5./12.]) + 0.1
# ours_grasping = np.array([1.]) + 0.1

# prior_task_success = np.array([0.]) + 0.1
# whirl_task_success = np.array([1./12.]) + 0.1
# ours_bo_task_success = np.array([4./12.]) + 0.1
# ours_task_success = np.array([11./12.]) + 0.1


# fig, ax = plt.subplots(1, 2)

# x = np.arange(len(tasks))  # the label locations
# width = 0.15  # the width of the bars
# multiplier = 0

# offset = width * multiplier
# rects = ax[0].bar(x + offset, prior_grasping, width, label="prior")
# # ax[0].bar_label(rects, padding=3)
# rects = ax[1].bar(x + offset, prior_task_success, width, label="prior")

# multiplier += 1
# offset = width * multiplier
# rects = ax[0].bar(x + offset, whirl_grasping, width, label="whirl")
# # ax.bar_label(rects, padding=3)
# rects = ax[1].bar(x + offset, whirl_task_success, width, label="whirl")

# multiplier += 1
# offset = width * multiplier
# rects = ax[0].bar(x + offset, ours_bo_grasping, width, label="ours-bo")
# # ax.bar_label(rects, padding=3)
# rects = ax[1].bar(x + offset, ours_bo_task_success, width, label="ours-bo")

# multiplier += 1
# offset = width * multiplier
# rects = ax[0].bar(x + offset, ours_grasping, width, label="ours")
# # ax.bar_label(rects, padding=3)
# rects = ax[1].bar(x + offset, ours_task_success, width, label="ours")


# # # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[0].set_ylabel('Success Rate')
# ax[0].set_xticks(x + width * 3/2, tasks)
# ax[0].legend(loc='upper left')
# ax[0].set_ylim([0, 1.2])
# ax[1].set_xticks(x + width * 3/2, tasks)
# ax[1].set_ylim([0, 1.2])
# ax[0].set_title("Grasping Success")
# ax[1].set_title("Task Success")
# fig.suptitle('multi Object Tasks', fontsize=16)
# plt.show()

# data = dict()

# task = dict()
# task["without_residual"] = [29, 10, 15]
# task["with_residual"] = [15, 11, 17]


task_1_without_res = [29, 10, 15]
task_2_without_res = [18, 33, 19]
task_3_without_res = [15, 11, 11]
task_4_without_res = [20, 20, 26]

task_1_with_res = [15, 11, 17]
task_2_with_res = [12, 20, 17]
task_3_with_res = [11, 11, 5]
task_4_with_res = [27, 29, 29]


averages_with_res = [np.mean(task_1_with_res), 
                     np.mean(task_2_with_res), 
                     np.mean(task_3_with_res), 
                     np.mean(task_4_with_res)]

yerr_with_res = [np.std(task_1_with_res)/np.sqrt(3), 
                 np.std(task_2_with_res)/np.sqrt(3),
                 np.std(task_3_with_res)/np.sqrt(3), 
                 np.std(task_4_with_res)/np.sqrt(3)]


averages_without_res = [np.mean(task_1_without_res), 
                     np.mean(task_2_without_res), 
                     np.mean(task_3_without_res), 
                     np.mean(task_4_without_res)]

yerr_without_res = [np.std(task_1_without_res)/np.sqrt(3), 
                 np.std(task_2_without_res)/np.sqrt(3),
                 np.std(task_3_without_res)/np.sqrt(3), 
                 np.std(task_4_without_res)/np.sqrt(3)]

tasks = ["task_1", "task_2", "task_3", "task_4"]

x = np.arange(len(tasks))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()

offset = width * multiplier
rects = ax.bar(x + offset, averages_without_res, width, label="without_residual", yerr=yerr_without_res)
ax.bar_label(rects, padding=3)
multiplier += 1
offset = width * multiplier
rects = ax.bar(x + offset, averages_with_res, width, label="with_residual", yerr=yerr_with_res)
ax.bar_label(rects, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Rollouts')
ax.set_xticks(x + width, tasks)
ax.legend(loc='upper left')

plt.show()