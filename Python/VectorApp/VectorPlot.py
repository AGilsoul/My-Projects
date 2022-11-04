from VectorUtils import Vector
from VectorUtils import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def create_empty_plot():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return fig


def create_plot_2d(V, origins):
    V = np.array(V)
    origins = np.array(origins).T
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.quiver(origins[0], origins[1], V[:, 0], V[:, 1], color=['r', 'g', 'b'], units='xy', angles='xy', scale_units='xy', scale=1)
    max_x = 2 * max(abs(V[:, 0]))
    max_y = 2 * max(abs(V[:, 1]))
    ax.set_xlim([-max_x, max_x])
    ax.set_ylim([-max_y, max_y])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return fig


def create_plot_3d(V, origins):
    V = np.array(V)
    origins = np.array(origins).T
    print(V)
    print(origins)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = colors.Normalize()
    norm.autoscale(len(V[0]))
    cmap = cm.get_cmap('Spectral')

    ax.quiver(origins[0], origins[1], origins[2], V[:, 0], V[:, 1], V[:, 2], cmap=cmap)
    max_x = 2 * max(abs(V[:, 0]))
    max_y = 2 * max(abs(V[:, 1]))
    max_z = 2 * max(abs(V[:, 2]))
    ax.set_xlim([-max_x, max_x])
    ax.set_ylim([-max_y, max_x])
    ax.set_zlim([-max_z, max_x])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig


def create_plot(V, origins=[]):
    if len(origins) == 0:
        origins = [[0 for _ in V[0]] for _ in V]
    if len(V[0]) == 2:
        return create_plot_2d(V, origins)
    if len(V[0]) == 3:
        return create_plot_3d(V, origins)
    raise Exception('Invalid Vector Dimensions')