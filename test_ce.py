import numpy as np
import yaml
from explorers import ExplorationPolicyCE

r_position = np.array([0.2543, 0.0021, 0.2])

def black_box_function(p):
    p = np.array(p)
    diff = np.abs(p-r_position)
    if diff[0] < 0.1 and diff[1] < 0.1 and diff[2] < 0.1: 
        return -100 * np.linalg.norm(p-r_position)
    if diff[0] < 0.05 and diff[1] < 0.05 and diff[2] < 0.05: 
        return -np.linalg.norm(p-r_position)
    else:
        return -1

cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

prior = np.zeros(3)

LIMITS = cfg["explorer_CE"]["LIMITS"]
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

explorer = ExplorationPolicyCE(id=1, prior=prior, limits=limits, env_size=3, savefolder="test", cfg=cfg) 

explorer.plot_centroids(name=1, subtask="0_approach", object_position=r_position)

trial = 0
while True:
    point = explorer.ask()
    reward = black_box_function(point)
    obj_wayp = np.zeros((2,3))
    if not reward == -1:
        obj_wayp[-1,:] = np.ones(3)

    explorer.tell(point, reward, obj_wayp)
    print("trial: {} point: {}, reward: {}".format(trial, point, reward))
    if reward > -0.01:
        break
    trial += 1
print(explorer.best_sample())