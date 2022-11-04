import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math



i_hat = np.array([1, 0, 0])
j_hat = np.array([0, 1, 0])
k_hat = np.array([0, 0, 1])


def update_vec(theta, m):
    return np.array([m*math.cos(theta), m*math.sin(theta), 0])


# return dot product of two vectors
def dot_product(u, v):
    return sum([u[x] * v[x] for x in range(len(u))])


# return vector orthogonal to two given vectors
def cross_product(u, v):
    return i_hat * (u[1] * v[2] - u[2] * v[1]) - j_hat * (u[0] * v[2] - u[2] * v[0]) + k_hat * (u[0] * v[1] - u[1] * v[0])


# return magnitude of a vector
def vec_magnitude(u):
    return math.sqrt(sum([x**2 for x in u]))


# return array of vector and orthogonal projection of v onto u
def vector_projection(u, v):
    c = dot_product(u, v) / vec_magnitude(u)**2
    w = c * u
    e = v - w
    return np.array([w, e])


vec_1 = np.array([4, 2, 7])
vec_2 = np.array([4, 0, 0])
m = vec_2[0]
print("Vec 1: ")
print(vec_1)
print("Vec 2: ")
print(vec_2)



def update_quiver(cur_theta, v1, m):
    global axis_quiver
    axis_quiver.remove()
    v2 = update_vec((cur_theta * 2 * math.pi)/360, m)
    projs = vector_projection(v1, v2)
    cp = cross_product(v1, v2)
    vec_matrix = np.array([v1, v2, projs[0], projs[1], cp]).T
    u = vec_matrix[0]
    v = vec_matrix[1]
    w = vec_matrix[2]
    num_vecs = len(vec_matrix.T)
    o_x = np.array([0 for _ in range(num_vecs)], dtype=float)
    o_y = np.array(o_x)
    o_z = np.array(o_x)
    o_x[3] = projs[0][0]
    o_y[3] = projs[0][1]
    o_z[3] = projs[0][2]
    axis_quiver = ax.quiver(o_x, o_y, o_z, u, v, w)
    return axis_quiver


# meow
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axis_quiver = ax.quiver([0], [0], [0], [0], [0], [0])
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

anim = animation.FuncAnimation(fig, update_quiver, fargs=(vec_1, m),
                               interval=25, blit=False)
fig.tight_layout()
plt.show()

