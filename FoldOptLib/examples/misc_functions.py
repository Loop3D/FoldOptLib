import numpy as np


def sample_random_dataset(grid, sample_size=2, seed=180):
    np.random.seed(seed)
    # select all xy points with z = 0
    xyz = grid[grid[:, 2] == 0]
    xmax = xyz[:, 0].max()
    ymax = xyz[:, 1].max()
    zmax = 0.
    xn = np.random.uniform(low=0, high=xmax, size=sample_size)
    yn = np.random.uniform(low=0, high=ymax, size=sample_size)
    zn = np.tile(zmax, sample_size)
    random_xyz = np.array([xn, yn, zn]).T

    return random_xyz
