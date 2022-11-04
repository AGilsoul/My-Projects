import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math
from Vector import *
from quaternions import *


i_hat = np.array([1, 0, 0])
j_hat = np.array([0, 1, 0])
k_hat = np.array([0, 0, 1])


def r_helix(theta, m):
    return m*math.cos(theta), m*math.sin(theta), theta


def v_helix(theta, m):
    return -m*math.sin(theta), m*math.cos(theta), 1


def a_helix(theta, m):
    return -m*math.cos(theta), -m*math.sin(theta), 0


def T_helix(theta, m):
    v1, v2, v3 = v_helix(theta, m)
    mag = Vector.vec_magnitude([v1, v2, v3])
    return v1/mag, v2/mag, v3/mag


def unit_normal_helix(theta, m):
    n1, n2, n3 = a_helix(theta, m)
    mag = Vector.vec_magnitude([n1, n2, n3])
    return n1/mag, n2/mag, n3/mag


def update_quiver_helix(cur_theta, m):
    global axis_quiver
    axis_quiver.remove()
    v_x, v_y, v_z = v_helix(cur_theta, m)
    t_x, t_y, t_z = unit_normal_helix(cur_theta, m)
    a_x, a_y, a_z = a_helix(cur_theta, m)
    o_x, o_y, o_z = r_helix(cur_theta, m)
    vector_matrix = np.array([[v_x, t_x, a_x], [v_y, t_y, a_y], [v_z, t_z, a_z]])
    x = vector_matrix[0]
    y = vector_matrix[1]
    z = vector_matrix[2]
    axis_quiver = ax.quiver(o_x, o_y, o_z, x, y, z)
    return axis_quiver


def r_loop(theta, m):
    return math.cos(1-0.1*theta), math.sin(0.1*theta), math.cos(0.1*theta)*math.sin(0.1*theta)


def v_loop(theta, m):
    return 0.1*math.sin(1-0.1*theta), 0.1*math.cos(0.1*theta), 0.1*math.cos(0.2*theta)


def a_loop(theta, m):
    return -(0.1**2)*math.cos(1-0.1*theta), -(0.1**2)*math.sin(0.1*theta), -0.02*math.sin(0.2*theta)


def T_loop(theta, m):
    v1, v2, v3 = v_loop(theta, m)
    mag = Vector.vec_magnitude([v1, v2, v3])
    return v1/mag, v2/mag, v3/mag


def unit_normal_loop(theta, m):
    n1, n2, n3 = a_loop(theta, m)
    mag = Vector.vec_magnitude([n1, n2, n3])
    return n1/mag, n2/mag, n3/mag


def update_quiver_loop(cur_theta, m):
    global axis_quiver
    axis_quiver.remove()
    v_x, v_y, v_z = v_loop(cur_theta, m)
    t_x, t_y, t_z = unit_normal_loop(cur_theta, m)
    a_x, a_y, a_z = a_loop(cur_theta, m)
    o_x, o_y, o_z = r_loop(cur_theta, m)
    # vector_matrix = np.array([[v_x, t_x, a_x], [v_y, t_y, a_y], [v_z, t_z, a_z]])
    vector_matrix = np.array([[v_x], [v_y], [v_z]])
    x = vector_matrix[0]
    y = vector_matrix[1]
    z = vector_matrix[2]
    axis_quiver = ax.quiver(o_x, o_y, o_z, x, y, z)
    return axis_quiver


p_o = np.array([3, 5, 2])


def vector_rotate(cur_theta):
    global axis_quiver
    axis_quiver.remove()
    rotated = Quaternion.rotate(p_o, j_hat, cur_theta)
    vector_matrix = rotated.T
    o_x = [0]
    o_y = [0]
    o_z = [0]
    x = vector_matrix[0]
    y = vector_matrix[1]
    z = vector_matrix[2]
    axis_quiver = ax.quiver(o_x, o_y, o_z, x, y, z)
    return axis_quiver


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axis_quiver = ax.quiver([0], [0], [0], [0], [0], [0])
lim = 10
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([0, 2 * lim])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# anim = animation.FuncAnimation(fig, update_quiver_loop, fargs=[3],
#                               interval=25, blit=False)
anim = animation.FuncAnimation(fig, vector_rotate, fargs=[],
                               interval=25, blit=False)
fig.tight_layout()
plt.show()
