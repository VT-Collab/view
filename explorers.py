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
import json
import pickle

'''    Exploration scheme using BO    '''
class ExplorationPolicyBO(object):
    def __init__(self, n_dims, limits, id, cfg, savefolder):
        kind = cfg['explorer_BO']['UTILITY']
        kappa = cfg['explorer_BO']['KAPPA']
        xi = cfg['explorer_BO']['XI']
        reward_threshold = cfg['training']['R_THRESH_BO']
        kappa_decay = cfg['explorer_BO']['K_DECAY']
        kappa_decay_delay = cfg['explorer_BO']['K_DECAY_DELAY']
        
        self.id = id
        
        self.n_dims = n_dims

        pbounds = self.list2param(limits)
        self.pbounds = pbounds
        self.reward_threshold = reward_threshold

        self.optimizer = BayesianOptimization(f=None, pbounds=pbounds, verbose=2, allow_duplicate_points=True)
        self.utility = UtilityFunction(kind=kind, kappa=kappa, xi=xi, 
                                       kappa_decay=kappa_decay, kappa_decay_delay=kappa_decay_delay)

        self.done = False
        self.buffer = dict()
        self.buffer["rollouts"] = []
        self.buffer["rewards"] = []
        
        self.savefolder = os.path.join(cfg['save_data']['DEBUG'], savefolder)
        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)
        
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
        best_sample, best_reward = self.best_sample()
        if best_reward > self.reward_threshold:
            self.done = True
            print(f'Explorer {self.id} converged at R = {best_reward}')
            return best_sample
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
        self.buffer["rollouts"].append(point)
        self.buffer["rewards"].append(reward)
        
        saveloc = os.path.join(self.savefolder, "explorer_{}_res.json".format(self.id))
        json.dump(self.buffer, open(saveloc, "w"))

    def best_sample(self):
        best_params = self.optimizer.max
        if len(best_params.keys()) == 0:
            return None, -np.inf
        waypoint = np.array(self.param2list(best_params["params"]))
        reward = best_params["target"]
        return waypoint, reward
        
    def plot_limits(self, name, subtask, demo_dir, cfg, gui=False):
        # load object position for reset and debug
        subtask_num = re.findall(r'\d+', subtask)[0]
        obj_path = os.path.join(cfg['save_data']['DEMOS'], demo_dir, 'object_positions.json')
        all_object_positions = json.load(open(obj_path, 'r'))    
        object_position = all_object_positions[str(subtask_num)]
                
        prior = np.array(json.load(open(os.path.join(
            self.savefolder, 'current_prior_{}.json'.format(subtask)), 'r')))
        limit_x = cfg['env']['LIMIT_X']
        limit_y = cfg['env']['LIMIT_Y']
        limit_z = cfg['env']['LIMIT_Z']

        xlim, ylim, zlim = self.param2list(self.pbounds)

        verts_coords = np.array(np.meshgrid(xlim, ylim, zlim)).T.reshape((-1, 3)).tolist()
        verts_indxs = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [2, 0, 4 ,6], [3, 1, 5, 7]]
        verts = [[verts_coords[verts_indxs[ix][iy]] for iy in range(len(verts_indxs[0]))] for ix in range(len(verts_indxs))]
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
        ax.add_collection3d(Poly3DCollection(verts, facecolors='b', alpha=0.1))
        ax.plot(object_position[0], object_position[1], object_position[2], 'g.', markersize=10)
        ax.plot(prior[:, 1], prior[:, 2], prior[:, 3], 'k:')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(tuple(limit_x))
        ax.set_ylim(tuple(limit_y))
        ax.set_zlim(tuple(limit_z))
        ax.view_init(30, -150)
        
        if gui:
            plt.show()
            exit()
        else:
            plt.savefig(os.path.join(self.savefolder, 'centroids_{}_{}.png'.format(subtask, name)), dpi=300)
            pickle.dump(fig, open(os.path.join(self.savefolder, 'centroids_fig_{}_{}.pickle'.format(subtask, name)), 'wb'))
        plt.close()
