import numpy as np
from explorers import ExplorationPolicyBO
import json
import yaml


def black_box_function(p):
    r_position = np.array([0.1543, 0.0021, 0.2])
    p = np.array(p)
    return -np.linalg.norm(p-r_position)

cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

prior = np.zeros(3)

LIMITS = 0.25
limit_x = [-0.5, 0.5]
limit_y = [-0.5, 0.5]
limit_z = [-0.5, 0.5]

lower_limits = prior.copy()
lower_limits -= LIMITS

lower_limits[0] = np.clip(lower_limits[0], limit_x[0], limit_x[1])
lower_limits[1] = np.clip(lower_limits[1], limit_y[0], limit_y[1])
lower_limits[2] = np.clip(lower_limits[2], limit_z[0], limit_z[1])

upper_limits = prior.copy()
upper_limits += LIMITS

upper_limits[0] = np.clip(upper_limits[0], limit_x[0], limit_x[1])
upper_limits[1] = np.clip(upper_limits[1], limit_y[0], limit_y[1])
upper_limits[2] = np.clip(upper_limits[2], limit_z[0], limit_z[1])

print("\n[*] Received prior\n {}".format(prior))
print("\n[*] Applying the following lower limits\n {}".format(lower_limits))
print("\n[*] Applying the following upper limits\n {}".format(upper_limits))

limits = np.column_stack((lower_limits.flatten(), upper_limits.flatten()))
limits = limits.tolist()
explorers = []
for i in range(3):
    explorer = ExplorationPolicyBO(n_dims=3, limits=limits, id=1, cfg=cfg, savefolder="test")
    explorers.append(explorer)


trial = 0
while True:
    for expl_num, explorer in enumerate(explorers):
        point = explorer.ask()
        reward = black_box_function(point)
        explorer.tell(point.tolist(), reward)
        print("explorer: {} trial: {} point: {}, reward: {}".format(expl_num, trial, point, reward))
    dones = [explorer.done for explorer in explorers]
    if all(dones):
        break
    trial += 1

# print(explorer.best_sample())
