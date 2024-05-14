import math
import random
import json
import numpy as np
import torch
# import pygame
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d
from scipy.stats import beta
import time
import os
import logging
import warnings
warnings.simplefilter("ignore")

"""
TODO
- 
"""
np.set_printoptions(precision=4, suppress=True)
plt.rcParams["font.size"] = 18


logger = logging.getLogger()

'''    Squeeze the prior to have fewer waypoints    '''
class SQUISH_E(object):
    def SED(self, p, p_pred, p_succ):
        pos_a = p_pred[1:4]
        t_a = p_pred[0]
        grip_a = p_pred[-1]
        pos_b = p[1:4]
        t_b = p[0]
        grip_b = p[-1]
        pos_c = p_succ[1:4]
        t_c = p_succ[0]
        grip_c = p_succ[-1]

        v = (pos_c - pos_a) / (t_c - t_a)
        pos_proj = pos_a + v * (t_b - t_a)
        
        v_grip = (grip_c - grip_a) / (t_c - t_a)
        grip_proj = grip_a + v_grip * (t_b - t_a)
        return np.linalg.norm(pos_b - pos_proj) + 0.003 * np.linalg.norm(grip_b - grip_proj)

    def adjust_priority(self, i, traj, Q, pred, succ, pi):
        if pred[i] > -1 and succ[i] > -1:
            Q[i] = pi[i] + self.SED(traj[i], traj[int(pred[i])], traj[int(succ[i])])
        return traj, Q, pred, succ, pi

    def reduce(self, traj, Q, pred, succ, pi):
        j = int(np.nanargmin(Q))

        pi[int(succ[j])] = max(Q[j], pi[int(succ[j])])
        pi[int(pred[j])] = max(Q[j], pi[int(pred[j])])
        succ[int(pred[j])] = succ[j]
        pred[int(succ[j])] = pred[j]
        traj, Q, pred, succ, pi = self.adjust_priority(int(pred[j]), traj, Q, pred, succ, pi)
        traj, Q, pred, succ, pi = self.adjust_priority(int(succ[j]), traj, Q, pred, succ, pi)
        Q[j] = np.nan
        # print(j, Q[j])
        return traj, Q, pred, succ, pi

    def squish(self, traj, lamda=0.5, mu=0, vers="nstd"):
        """
        traj: input ndarray of size nxd, where n is the number of points and d is the dimensionality
        lambda: compression ratio between output points and input points. Lies in [0, 1].
        mu: Maximum acceptable error in the system
        returns compressed_traj with size pxd, where p <= n * lamda
        """
        beta = np.ceil(len(traj) * lamda)
        n_pts = len(traj)
        Q = np.full(n_pts, np.infty)
        pi = np.ones(n_pts) * -1
        succ = np.ones(n_pts) * -1
        pred = np.ones(n_pts) * -1
        gripper_state = traj[:, -1]

        if vers == "std":
            for i in range(n_pts):
                Q[i] = np.infty
                pi[i] = 0

                if i >= 1:
                    succ[i - 1] = i
                    pred[i] = i - 1
                    traj, Q, pred, succ, pi = self.adjust_priority(
                        i - 1, traj, Q, pred, succ, pi
                    )
            q_non_zero = np.sum(np.invert(np.isnan(Q)))
            while q_non_zero > beta:
                traj, Q, pred, succ, pi = self.reduce(traj, Q, pred, succ, pi)
                q_non_zero = np.sum(np.invert(np.isnan(Q)))
        else:
            q_non_zero = np.sum(np.invert(np.isnan(Q)))
            while q_non_zero > beta:
                for i in range(n_pts):
                    pi[i] = 0

                    if i >= 1:
                        if np.isnan(Q[i]) or np.isnan(Q[i-1]):
                            continue
                        succ[i - 1] = i
                        pred[i] = i - 1
                        traj, Q, pred, succ, pi = self.adjust_priority(
                            i - 1, traj, Q, pred, succ, pi
                        )
                traj, Q, pred, succ, pi = self.reduce(traj, Q, pred, succ, pi)
                q_non_zero = np.sum(np.invert(np.isnan(Q)))
        p = np.nanmin(Q)
        while p <= mu:
            traj, Q, pred, succ, pi = self.reduce(traj, Q, pred, succ, pi)
            p = np.nanmin(Q)
        squished_traj = traj[np.invert(np.isnan(Q))]
        
        squished_indices = [np.where(traj[:, 0] == squished_traj[i, 0])[0][0]
                            for i in range(len(squished_traj))]
        squished_indices = squished_indices[1:-1]
        for i, s_idx in enumerate(squished_indices):
            start = 0 if i == 0 else squished_indices[i - 1]
            middle = s_idx
            end = len(traj)-1 if i == len(squished_indices)-1 else squished_indices[i + 1]
            prev_contact = np.sum(gripper_state[start:middle]) >= (middle - start)/1.5
            next_contact = np.sum(gripper_state[middle:end]) >= (end-middle)/1.5
            if prev_contact == 1 or next_contact == 1:
                squished_traj[i+1, -1] = 1
        return squished_traj
    
'''    Read joystick inputs    '''
class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False

        # some constants
        self.step_size_l = 0.15
        self.step_size_a = 0.2 * np.pi / 4
        self.step_time = 0.01
        self.deadband = 0.1

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [-z1, z2, -z3]
        for idx in range(len(z)):
            if abs(z[idx]) < self.deadband:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        X_pressed = self.gamepad.get_button(2)
        B_pressed = self.gamepad.get_button(1)
        A_pressed = self.gamepad.get_button(0)
        return tuple(z), A_pressed, B_pressed, X_pressed, stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, self.step_size_a * -z[1], self.step_size_a * -z[0], self.step_size_a * -z[2])
        else:
            self.action = (self.step_size_l * -z[1], self.step_size_l * -z[0], self.step_size_l * -z[2], 0, 0, 0)

'''    Calculate rewards for task    '''
class ManipulateReward(object):

    def __init__(self, demo_folder):
        obj_filename = os.path.join(demo_folder, "obj_traj.json")
        with open(obj_filename, "r") as f:
            demo_obj_traj = np.array(json.load(f))
        self.demo_obj_traj = Trajectory(demo_obj_traj)

    def waypoint_reward(self, t, rollout_obj_pos):
        demo_obj_pos = self.demo_obj_traj.get_waypoint(t)
        return -np.linalg.norm(demo_obj_pos - rollout_obj_pos)

    def compute_reward(self, traj, times):
        rollout_obj_traj = Trajectory(traj)
        reward = []
        for i in range(len(times)):
            t = times[i]
            rollout_waypoint = rollout_obj_traj.get_waypoint(t)
            reward.append(self.waypoint_reward(t, rollout_waypoint))
        return reward

'''    Memory buffer    '''
class Buffer(object):
    def __init__(self, n_waypoints):
        # self.capacity = capacity
        self.n_waypoints = n_waypoints
        self.waypoints = dict()
        self.rewards = dict()
        for i in range(self.n_waypoints):
            key = str(i)
            self.waypoints[key] = []
            self.rewards[key] = []
        self.buffer_size = 0

    def push(self, data, rewards):
        # data is a set of waypoints with dims n_waypoints x N
        # Here N is the dimension for each waypoint (ex: [x,y,z] or [x,y,z,r,p,y])
        # rewards are the rewards for each individual waypoint with dims n_waypoints x 1

        data = data.tolist()
        for i in range(self.n_waypoints):
            key = str(i)
            self.waypoints[key].append(data[i])
            self.rewards[key].append(rewards[i])
        self.buffer_size = len(self.waypoints["0"])

        logging.debug("residuals pushed. Buffer has {} datapoints ".format(self.buffer_size))

    def best_sample(self):
        # Find the best waypoint for each location in the trajectory
        if self.buffer_size == 0:
            return None
        else:
            best_waypoints = []
            best_rewards = []
            for i in range(self.n_waypoints):
                key = str(i)
                sort_idxs = np.argsort(self.rewards[key])
                best_waypoints.append(self.waypoints[key][sort_idxs[-1]])
                best_rewards.append(self.rewards[key][sort_idxs[-1]])
            sample = np.array(best_waypoints)
            rewards = np.array(best_rewards)
            return sample, rewards

    def sample(self, sample_size, best_n=20):
        # picks the top best_n waypoints for each location
        # samples from them using sample_size

        if self.buffer_size < best_n:
            best_n = self.buffer_size

        waypoints_subset = dict()
        rewards_subset = dict()
        for i in range(self.n_waypoints):
            key = str(i)
            sort_idxs = np.argsort(self.rewards[key])
            waypoints_subset[key] = [self.waypoints[key][idx] for idx in sort_idxs[-best_n:]]
            rewards_subset[key] = [self.rewards[key][idx] for idx in sort_idxs[-best_n:]]

        samples = []
        for sample_num in range(sample_size):
            sample = []
            for i in range(self.n_waypoints):
                key = str(i)
                waypoints = waypoints_subset[key]
                rewards = rewards_subset[key]
                p = np.exp(rewards) / sum(np.exp(rewards))
                # print(rewards)
                # p = np.array(rewards) / np.sum(rewards)
                # print(p)
                sample_idx = np.random.choice(range(len(rewards)), p=p)
                sample.append(waypoints[sample_idx])
                logging.debug("Sample for sample num {} and position {} sampled using weights {}".format(sample_num, i, p))
            samples.append(np.array(sample))

        return samples

    def __len__(self):
        return self.buffer_size

'''    Index with largest rewards    '''
def get_largest_indices(arr, n):
    """Returns the n largest indices from a numpy array."""
    flat = arr.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]

    return np.unravel_index(indices, arr.shape)

'''    formatter for logging    '''
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = ("%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)")

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
class Trajectory():

    def __init__(self, traj):
        self.traj = np.copy(traj)
        self.n, self.m = traj.shape
        self.t_start = traj[0,0]
        self.t_end = traj[-1, 0]
        self.total_time = self.t_end - self.t_start
        self.times = traj[:, 0]
        
        self.has_grip = False
        if self.m == 5:
            self.has_grip = True
        #     # gripper is only on/off so we searchsort for action of nearest lower bound
            self.gripper_state = traj[:,-1]   

        self.interpolators = []
        for idx in range(self.m):
            self.interpolators.append(interp1d(self.times, self.traj[:, idx], kind='linear'))
        
    def get_gripper_action(self, sample_time):
        if sample_time in self.times:
            closest_gripper_idx = np.where(self.times==sample_time)
        else:
            closest_gripper_idx = np.searchsorted(self.times, sample_time) - 1
            closest_gripper_idx = np.clip(closest_gripper_idx, 0, len(self.times))
        
        # try: 
        #     closest_gripper_idx = closest_gripper_idx[-1]
        # except:
        #     pass

        return self.gripper_state[closest_gripper_idx]    
    
    def get_waypoint(self, t):
        if t < 0.0:
            t = 0.0
        if t > self.t_end:
            t = self.t_end
        waypoint = np.array([0.] * self.m)
        for idx in range(self.m):
            waypoint[idx] = self.interpolators[idx](t)
        if self.has_grip: # has gripper state
            waypoint[-1] = self.get_gripper_action(t)
        return waypoint[1:]