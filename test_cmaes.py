import numpy as np
import logging
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
from scipy.spatial.distance import cdist
from sklearn.cluster import k_means
import re
from copy import copy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import yaml, json
from cmaes import CMA

'''    Exploration scheme using BO    '''
class ExplorationPolicyBO(object):
    def __init__(self, n_dims, limits):
        kind = 'ucb'
        kappa = 1e-2
        xi = 1e-1
        kappa_decay = 0.3
        kappa_decay_delay = 0
                
        self.n_dims = n_dims
        pbounds = self.list2param(limits)
        self.pbounds = pbounds

        self.optimizer = BayesianOptimization(f=None, pbounds=pbounds, verbose=2, allow_duplicate_points=True)
        self.utility = UtilityFunction(kind=kind, kappa=kappa, xi=xi, 
                                       kappa_decay=kappa_decay, kappa_decay_delay=kappa_decay_delay)

        self.done = False
        
        
    def param2list(self, params):
        l = np.array([params[str(i)] for i in range(self.n_dims)])
        return l

    def list2param(self, l):
        d = dict()
        for i in range(self.n_dims):
            d[str(i)] = l[i]
        return d

    def ask(self):
        np.random.seed()
        suggestion = self.optimizer.suggest(self.utility)
        waypoint = np.array(self.param2list(suggestion))
        
        # update the k for ucb
        self.utility.update_params()
        # print('K: ', self.utility.kappa)
        
        return waypoint

    def tell(self, point, reward):
        if self.done:
            return None
        point_params = self.list2param(point)
        self.optimizer.register(params=point_params,target=reward)

    def best_sample(self):
        best_params = self.optimizer.max
        if len(best_params.keys()) == 0:
            return None, -np.inf
        waypoint = np.array(self.param2list(best_params["params"]))
        reward = best_params["target"]
        return waypoint, reward
        
class ExplorationPolicyCMAES(object):
    def __init__(self, prior, limits, n_suggestions=5, threshold=0):

        self.prior = prior
        self.sigma = 0.25
        self.done = False
        self.population_size = n_suggestions
        self.threshold = threshold
        self.optimizer = CMA(self.prior, self.sigma, np.array(limits),)# population_size=self.population_size, lr_adapt=True)
        
        self.suggestion_number = 0
        self.all_suggestions = []
        self.all_rewards = []

        self.suggestions = []
        self.asked = False
        self.done = False

    def ask(self):
        if self.done:
            return self.best_sample()

        assert self.asked == False, "You have already asked. Must tell before asking again."
        np.random.seed()
        waypoint = self.optimizer.ask()
        self.asked = True
        return waypoint

    def tell(self, point, reward):
        if self.done:
            return
        assert self.asked==True, "You must ask before telling."
        cost = -reward
        self.all_suggestions.append(point)
        self.all_rewards.append(reward)

        if reward > self.threshold:
            self.done = True

        if len(self.suggestions) < self.population_size:
            self.suggestions.append((point, cost))
        else:
            self.optimizer.tell(self.suggestions)
            self.suggestions = []

        self.asked = False

    def best_sample(self):
        idx = np.argmax(self.all_rewards)
        return self.all_suggestions[idx]

if __name__ == "__main__":

    r_position = np.array([0.2543, 0.0021, 0.2])

    def black_box_function(p):
        p = np.array(p)
        diff = np.abs(p-r_position)
        return -np.linalg.norm(p-r_position)
        # if diff[0] < 0.1 and diff[1] < 0.1 and diff[2] < 0.1: 
        #     return -100 * np.linalg.norm(p-r_position)
        # if diff[0] < 0.05 and diff[1] < 0.05 and diff[2] < 0.05: 
        #     return -np.linalg.norm(p-r_position)
        # else:
        #     return -1
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

    prior = np.zeros(3)

    LIMITS = 1.0
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

    explorer = ExplorationPolicyCMAES(prior, limits)

    trial = 0
    while True:
        point = explorer.ask()
        reward = black_box_function(point)
        explorer.tell(point, reward)
        print("trial: {} point: {}, reward: {}".format(trial, point, reward))
        if reward > -0.01:
            break
        trial += 1
    print(explorer.best_sample())