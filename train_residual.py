
import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
import torch
from nn_utils import ResidualNet, train_net
import os
import warnings
warnings.simplefilter("ignore")

device = "cpu"

def distort_coords(x, limits):
    coords = x.copy()
    mu = [np.mean(limits[0]), np.mean(limits[1]), np.mean(limits[2])]
    noise = np.tanh( (x-mu) / 2)
    return coords + noise


if __name__ == "__main__":
    cfg = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)


    limit_x = cfg['env']['LIMIT_X']
    limit_y = cfg['env']['LIMIT_Y']
    limit_z = cfg['env']['LIMIT_Z']
    lower_limits = [limit_x[0], limit_y[0], limit_z[0]]
    upper_limits = [limit_x[1], limit_y[1], limit_z[1]]

    # get initial points
    total_points = 100
    data = np.random.uniform(lower_limits, upper_limits, (total_points, 3))
    indexes = np.arange(total_points)
    np.random.shuffle(indexes)
    
    training_idxs = indexes[:int(0.8*total_points)]
    test_idxs = indexes[int(0.8*total_points):]
    gt = data[training_idxs, :]
    test_data = data[test_idxs, :]
    limits = [limit_x, limit_y, limit_z]
    distorted_data = distort_coords(gt, limits)

    epochs = 100
    batch_size = len(gt)
    lr = 0.1
    lr_step_size = 400
    lr_gamma = 0.15

    dataset = []
    for idx in range(len(gt)):
        dataset.append([distorted_data[idx,:].tolist(), gt[idx,:].tolist()])

    savename = "model.pkl"

    train_net(dataset, epochs, lr, batch_size, lr_step_size, lr_gamma, savename)

    savename = os.path.join("./models", savename)
    model = ResidualNet().to(device)
    model_dict = torch.load(savename, map_location=device)
    model.load_state_dict(model_dict)
    model.eval()

    test_data_distorted = distort_coords(test_data, limits)
    test_data_distorted = torch.FloatTensor(test_data_distorted)
    output = model.get_residual(test_data_distorted).detach().numpy()
    error = np.mean(np.linalg.norm(test_data-output, axis=1))
    print("Test set error: {}".format(error))
