import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


i_hat = np.array([1, 0, 0])
j_hat = np.array([0, 1, 0])
k_hat = np.array([0, 0, 1])


# return dot product of two vectors
def dot_product(u, v):
    return sum([u[x] * v[x] for x in range(len(u))])


def cross_product(u, v):
    return i_hat * (u[1] * v[2] - u[2] * v[1]) + j_hat * (u[0] * v[2] - u[2] * v[0]) + k_hat * (u[0] * v[1] - u[1] * v[0])


# return magnitude of a vector
def vec_magnitude(u):
    return math.sqrt(sum([x**2 for x in u]))


# return array of vector and orthogonal projection of v onto u
def vector_projection(u, v):
    c = dot_product(u, v) / vec_magnitude(u)**2
    w = c * u
    e = v - w
    return np.array([w, e])



vec_1 = np.array([5, 0, 0])
vec_2 = np.array([3, 3, 0])
projs = vector_projection(vec_1, vec_2)
cp = cross_product(vec_1, vec_2)
vector_matrix = np.array([vec_1, vec_2, projs[0], projs[1], cp]).T
print(vector_matrix)

U = [x[0] for x in vector_matrix]
V = [y[1] for y in vector_matrix]
W = [z[2] for z in vector_matrix]

# meow
origins_x = np.array([0 for x in range(len(vector_matrix[0]))])
origins_y = np.array([0 for y in range(len(vector_matrix[0]))])
origins_z = np.array([0 for z in range(len(vector_matrix[0]))])
origins_x[3] = projs[0][0]
origins_y[3] = projs[0][1]
origins_z[3] = projs[0][2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
norm = colors.Normalize()
norm.autoscale(len(vector_matrix[0]))
cmap = cm.get_cmap('Spectral')

ax.quiver(origins_x, origins_y, origins_z, vector_matrix[0], vector_matrix[1], vector_matrix[2], cmap=cmap)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()