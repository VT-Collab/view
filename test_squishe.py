from utils import SQUISH_E, Trajectory
import numpy as np
import random
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

obj_position = [0.4, -0.3, 0.]

start = [0.4, -0.3, 0.45, 0]
pickup = obj_position + [1]
end = [0.5, -0.2, 0.45, 1]

times = [0., 1., 2.]

waypoints = np.array([start, pickup, end])
traj = np.column_stack(((times, waypoints)))

fig, axs = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(15,15))
axs.plot(traj[:, 1], traj[:, 2], traj[:, 3], 'ko', label='Given Waypoints')

traj_fn = Trajectory((traj))

dt = 0.0001
t = 0
max_t = 2.

full_traj = []
while t < max_t:
    waypoint = traj_fn.get_waypoint((t))
    full_traj.append([t] + waypoint.tolist())
    t += dt
full_traj = np.array(full_traj)
axs.plot(full_traj[:, 1], full_traj[:, 2], full_traj[:, 3], 'r', label='Original Traj')

compressor = SQUISH_E()

compressed_traj = compressor.squish(full_traj, lamda=1.0, mu=1e-2)
axs.plot(compressed_traj[:, 1], compressed_traj[:, 2], compressed_traj[:, 3], 'bx', label='Squished Waypoints')

print("Found {} points after compressing original trajectory".format(compressed_traj.shape[0]))
print("Compressed Traj: \n {}".format(compressed_traj))

gaussian_traj = compressed_traj.copy()
mu = 0.02 
sigma = 0.01
gaussian_traj[1:,1:-1] += np.random.normal(mu, sigma, size=gaussian_traj[1:,1:-1].shape)
axs.plot(gaussian_traj[:, 1], gaussian_traj[:, 2], gaussian_traj[:, 3], 'c', label='Gaussian Waypoints mu {} sigma {}'.format(mu, sigma))
print("Distorted Traj: \n {}".format(gaussian_traj))

axs.legend()
plt.show()
