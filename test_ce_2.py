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
import pickle

'''    Exploration scheme using BO    '''
class ExplorationPolicyBO(object):
    def __init__(self, n_dims, limits):
        kind = 'ucb'
        kappa = 10
        xi = 1e-1
        kappa_decay = 1.0
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
        
class ExplorationPolicyCE(object):
    def __init__(self, id, limits, prior, env_size, cfg, savefolder, verbose=False, samples=1e5):
        
        self.id = id
        
        self.prior = np.array(prior).squeeze()
        self.n_dims = len(self.prior)
        self.env_size = env_size
        self.cells = cfg['explorer_CE']['N_CELLS']
        self.sigma = cfg['explorer_CE']['SIGMA']
        self.alpha = cfg['explorer_CE']['ALPHA']
        self.gamma = cfg['explorer_CE']['GAMMA']
        self.p_explore = cfg['explorer_CE']['P_EXPLORE']
        self._verbose = verbose
        self.samples = int(samples)
        
        limits = list(zip(*limits))
        self._lower_limits = list(limits[0])
        self._upper_limits = list(limits[1])
        
        self.savefolder = os.path.join(cfg['save_data']['DEBUG'], savefolder)
        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)
        
        self.visited_centroid_idxs = []
        self.unvisited_centroid_idxs = [i for i in range(self.cells)]
        self.rewards = []
        self.best_solution = np.zeros((self.cells, self.n_dims))
        self._unvisited_entropy = None
        
        self.done = False
        self.reward_threshold_percent = -0.01
        
        self.all_suggestions = []
        self.all_rewards = []
        
        if self._verbose:
            self.counter = 0
            print('limits:\n ', limits)
            
        self._generate_centroids()
   
        if self._verbose:
            print('centroids:\n ', self._centroids)
        
        self.local_optimizer = None
        
    def _generate_centroids(self):
        rng = np.random.default_rng(None)
        
        if self._verbose:
            print('\nGenerating centroids...\n')
        self._samples = rng.uniform(self._lower_limits, 
                                    self._upper_limits, 
                                    size=(self.samples, self.n_dims)).astype(np.float64)
        self._centroids = k_means(self._samples, self.cells - 1, init='random')[0]
        
        # add prior to the list of centroids
        self._centroids = np.vstack((self.prior[None, :], self._centroids))

        assert self._centroids.shape[0] == self.cells,"k-means clustering found {} centroids,\
        but the explorer needs {} centroids.".format(self._centroids.shape[0], self.cells)
    
    def plot_centroids(self, name, subtask, object_position=None, gui=False):
        print('[*] Plotting centroids...')
        centroids_plotting = self._centroids.reshape((self.cells, -1, self.env_size))
        prior = self.prior.copy()
        prior = prior.reshape((-1, self.env_size))
        
        if gui:
            plt.switch_backend('TkAgg')
        else:
            plt.switch_backend('Agg') 
        fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))
        x = centroids_plotting[:, :, 0].flatten()
        y = centroids_plotting[:, :, 1].flatten()
        z = centroids_plotting[:, :, 2].flatten()
        
        if not object_position is None:
            # find number of centroids within epsilon-ball of object
            object_position = np.array(object_position)
            distances = np.linalg.norm(centroids_plotting.squeeze() - object_position, axis=1)
            num_closeby_5 = np.sum(distances <= 0.05)
            num_closeby_3 = np.sum(distances <= 0.03)
            axs.plot(object_position[0], object_position[1], object_position[2], 'g.', markersize=10)
        
        axs.plot(prior[:, 0], prior[:, 1], prior[:, 2], 'k:')
        axs.plot(x, y, z, 'bx', markersize=2, alpha=0.2)
        axs.set_xlabel("X")
        axs.set_ylabel("Y")
        axs.set_zlabel("Z")
        axs.set_xlim(self._lower_limits[0], self._upper_limits[0])
        axs.set_ylim(self._lower_limits[1], self._upper_limits[1])
        axs.set_zlim(self._lower_limits[2], self._upper_limits[2])
        axs.set_title('within 3 cm: {}, within 5 cm: {}'.format(num_closeby_3, num_closeby_5))
        axs.view_init(30, -150)
        
        if gui:
            plt.show()
            exit()
        else:
            pickle.dump(fig, open(os.path.join(self.savefolder, 'centroids_fig_{}_{}.pickle'.format(subtask, name)), 'wb'))
            plt.savefig(os.path.join(self.savefolder, 'centroids_{}_{}.png'.format(subtask, name)), dpi=300)
        plt.close()
        
    def _annealing(self):
        if self.alpha is not None:
            R = np.array(copy(self.rewards))
            # mu = self.rewards[0]
            mu = np.median(self.rewards)
            max_variance = np.max((R - mu) ** 2 / np.abs(mu))
            self.p_explore = np.tanh(self.alpha / max(max_variance, 1e-6))
            if self._verbose:
                print("Rewards: {}".format(R))
            print('p {}, m_var: {}, mu: {}'.format(self.p_explore, max_variance, mu))
        
    def _update(self, tried_rollout, reward, obj_wayp):
        self.all_suggestions.append(tried_rollout[0])
        self.all_rewards.append(reward)
        
        if self._verbose:
            print("all suggestions \n {} \n all_rewards \n {}".\
                format(np.array(self.all_suggestions), np.array(self.all_rewards)))
        # update the internal memory if you visit new centroids
        if self.going2new_centroids:
            self.unvisited_centroid_idxs.remove(self._suggestion_ind)
            self.visited_centroid_idxs.append(self._suggestion_ind)
            self.rewards.append(reward)
        
        else:
            # current_centroid_reward = self.rewards[self.visited_centroid_idxs.index(self._suggestion_ind)]
            # if self.local_optimizer is not None:
            self.local_optimizer.tell(tried_rollout, reward)
            # check to see if there is any movement in the cup position
            # obj_wayp = np.array(obj_wayp)
            # obj_movement = np.linalg.norm(obj_wayp[0,:] - obj_wayp[-1,:]) > 1
            # if reward > current_centroid_reward and obj_movement:
            #     print('\nUpdating the centroid {}'.format(self._suggestion_ind))
            #     self._centroids[self._suggestion_ind] = np.copy(tried_rollout)
            #     self.rewards[self.visited_centroid_idxs.index(self._suggestion_ind)] = reward
        
        if len(self.all_rewards) % 50 == 0:
            self._annealing()
        
        if self._verbose:
            print('updated visited:\n ', self.visited_centroid_idxs)
            print('updated unvisited:\n ', self.unvisited_centroid_idxs)
            print('updated rewards: \n', self.rewards)
    
    def _calculate_pdist(self):
        k = int(0.1 * self.cells)
        visited_points = self._centroids[self.visited_centroid_idxs, :]
        unvisited_points = self._centroids[self.unvisited_centroid_idxs, :]
        
        if len(self.visited_centroid_idxs) < k:
            k = len(self.visited_centroid_idxs)
        dist = cdist(visited_points, unvisited_points, metric='sqeuclidean')
        prior = self.prior.copy()
        if len(prior.shape) < 2:
            prior = prior[None, :]
        # dist = cdist(prior, unvisited_points, metric='sqeuclidean')
        knn_dist = np.mean(np.sort(dist, axis=0)[:k, :], axis=0)
        knn_dist = (knn_dist - np.min(knn_dist)) / max(np.max(knn_dist) - np.min(knn_dist), 1e-7)

        inv_var = 1 / (np.var(dist, axis=0) + 1)
        inv_var = (inv_var - np.min(inv_var)) / max(np.max(inv_var) - np.min(inv_var), 1e-7)

        # self._particle_dist = dist.copy()#knn_dist + inv_var
        self._particle_dist = knn_dist + inv_var
        
    def _calculate_entropy(self):
        self._calculate_pdist()
        self._unvisited_entropy = np.log(self._particle_dist)
        # self._unvisited_entropy = self._particle_dist
        
    def _entropy_explore(self):
        if len(self.visited_centroid_idxs) == 0:
            self._suggestion_ind = 0
        else:
            self._calculate_entropy()
            # print("Closest centroid dist: {}".format(np.min(self._unvisited_entropy)))
            self._suggestion_ind = self.unvisited_centroid_idxs[np.argmax(self._unvisited_entropy)]
            
    def _reward_explore(self):
        '''    based on change in reward    '''
        exp_prob = np.asarray(self.rewards)
        exp_prob = np.abs(exp_prob - self.rewards[0])
        exp_prob /= max(exp_prob)
        exp_prob_exp = np.exp(self.gamma*exp_prob) / np.sum(np.exp(self.gamma*exp_prob))
        # self._suggestion_ind = np.random.choice(self.visited_centroid_idxs, p=exp_prob_exp)
        self._suggestion_ind = self.visited_centroid_idxs[np.argmax(exp_prob_exp)]

        # if self._verbose:
        print("Explore prb before norm: {}".format(exp_prob))
        print("Explore prb after norm: {}".format(exp_prob_exp))

    def ask(self):
        if len(self.rewards) >= 1:
            self.reward_threshold = self.reward_threshold_percent * self.rewards[0]

            # check if converged
            best_sample, best_reward, best_sample_idx = self.best_sample()
            
            if self._verbose:
                print(f'best R, \n{best_reward}, \nrewards: \n{self.rewards}')
                print(f'best c, \n{best_sample}, \nsamples: \n{self._centroids[self.visited_centroid_idxs]}, \
                    \nbest idx: {best_sample_idx}')
            
            if best_reward > self.reward_threshold:
                self.done = True
                print(f'Explorer {self.id} converged at R = {best_reward}')
                return best_sample            
            
        if (np.random.uniform(0., 1.) < self.p_explore and len(self.unvisited_centroid_idxs) != 0) or len(self.visited_centroid_idxs) == 0:
            self.going2new_centroids = True
            self._entropy_explore()
        else:
            self.going2new_centroids = False
            self._reward_explore()
            if self._verbose:
                print('reward explore: \n', self.rewards)
        suggestion = None
        if not self.going2new_centroids:
            if self.local_optimizer is None:
                
                best_centroid = self._centroids[self._suggestion_ind].copy()
                lower_limits = best_centroid.copy() - 0.05

                lower_limits[0] = np.clip(lower_limits[0], self._lower_limits[0], self._upper_limits[0])
                lower_limits[1] = np.clip(lower_limits[1], self._lower_limits[1], self._upper_limits[1])
                lower_limits[2] = np.clip(lower_limits[2], self._lower_limits[2], self._upper_limits[2])

                upper_limits = best_centroid.copy() + 0.05

                upper_limits[0] = np.clip(upper_limits[0],self._lower_limits[0], self._upper_limits[0])
                upper_limits[1] = np.clip(upper_limits[1],self._lower_limits[1], self._upper_limits[1])
                upper_limits[2] = np.clip(upper_limits[2],self._lower_limits[2], self._upper_limits[2])
                limits = np.column_stack((lower_limits.flatten(), upper_limits.flatten()))
                limits = limits.tolist()
                print("Applying the following limits: {}".format(limits))
                self.local_optimizer = ExplorationPolicyBO(self.n_dims, limits)
            # suggestion += np.random.normal(0, self.sigma, size=suggestion.shape)
            # suggestion = np.clip(suggestion, self._lower_limits, self._upper_limits)
                # print("here")
            # asking optimizer
            suggestion = self.local_optimizer.ask()
        else:
            suggestion = self._centroids[self._suggestion_ind].copy()
        
        # if self._verbose:
        print("p_explore: {}, Chosen suggestion: {}, idx: {}".format(self.p_explore, suggestion, self._suggestion_ind))
        
        return suggestion
        
    def tell(self, tried_rollout, reward, waypoints):
        self._update(tried_rollout, reward, waypoints)

    def best_sample(self):
        if len(self.rewards) == 0:
            return None, -np.inf, None
        best_reward = np.max(self.rewards)
        best_sample_idx = self.visited_centroid_idxs[np.argmax(self.rewards)]
        best_sample = self._centroids[best_sample_idx]
        return best_sample, best_reward, best_sample_idx

if __name__ == "__main__":

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