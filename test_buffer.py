import numpy as np
from utils import Buffer, load_prior, distort_prior, squish_traj, CAE, ExplorationPolicy
import torch
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
# plt.rcParams["font.size"] = 18


filename = "./demos/placing/demo.json"

prior_clean = load_prior(filename)
distorted_prior = distort_prior(prior_clean)
squished_prior = squish_traj(distorted_prior, lamda=10/len(distorted_prior))

# print(len(distorted_prior))

# fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))
# axs.plot(prior_clean[:,1], prior_clean[:,2], prior_clean[:,3], 'bx', label='Demos1')
# axs.plot(squished_prior[:,1], squished_prior[:,2], squished_prior[:,3], 'rx', label='Demos1')
# plt.show()

buffer = Buffer(len(squished_prior))

explorer = ExplorationPolicy(squished_prior, buffer)
explorer.plot_compressions()

for i in range(10):
    distorted_prior = distort_prior(prior_clean)
    squished_prior = squish_traj(distorted_prior, lamda=10/len(distorted_prior))
    squished_waypoints = squished_prior[:,1:]

    reward = np.ones(len(squished_prior)) * np.random.choice(np.arange(10))
    buffer.push(squished_waypoints, reward)

samples = buffer.sample(5)
print(samples)
print(squished_prior)
residuals = [sample - squished_waypoints for sample in samples]
residuals = np.concatenate(residuals, axis=0)
print(residuals)

# wayp = explorer.explore()
# print(squished_prior)
# print(wayp)

# print(prior_clean)
# print(distorted_prior)
# v1 = torch.FloatTensor(prior_clean)
# l, w = v1.shape
# z = torch.ones(len(v1))
# # print(z.shape)
# v2 = torch.column_stack((z,v1))
# model = CAE(input_dim=w)
# out = model.get_residual(v1)
# print(l, w)
# print(out)
# print(v1)
# print(v2)
# n_waypoints = prior_clean.shape[0]
# buffer = Buffer(n_waypoints=n_waypoints)
# for i in range(50):
#     # distorted_prior = distort_prior(prior_clean)
#     distorted_prior = np.ones(prior_clean.shape) * i
#     rewards = []
#     for _ in range(n_waypoints):
#         rewards.append(-i)
#     buffer.push(distorted_prior, rewards)
# # print(distorted_prior)
# print(buffer.best_sample())
# sample = buffer.sample(1)
# print(sample)
# # print(len(buffer))
