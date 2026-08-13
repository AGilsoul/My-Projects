import numpy as np
import sys
from NumericalMethods.eigen import eigen
from NumericalMethods.matrix import *
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

K = 5

def distance(p1, p2):
    if len(p1) != len(p2):
        raise Exception('Cannot compute distance between points with different dimensionality.')
    return np.sqrt(sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))

def adjacency_matrix(points: list[list[float]]) -> list[list[float]]:
    """Computes the adjacency matrix for a given set of points, uses k-nearest-neighbors.

    Args:
        points (list[list[float]]): Points to determine adjacency with.

    Returns:
        list[list[float]]: Adjacency matrix for the given points.
    """
    adj_matrix = [[0 for _ in range(len(points))] for _ in range(len(points))]
    # For every point
    for i in range(len(points)):
        distances = [distance(points[i], points[j]) if i != j else sys.maxsize for j in range(len(points))]
        min_indices = []
        # Get the next closest point, K times
        for k in range(K):
            cur_min = sys.maxsize
            cur_min_index = -1
            for j in range(len(points)):
                if distances[j] < cur_min and j not in min_indices:
                    cur_min = distances[j]
                    cur_min_index = j
            min_indices.append(cur_min_index)
        # now update adjacency matrix
        for index in min_indices:
            adj_matrix[i][index] = 1
            adj_matrix[index][i] = 1
    return adj_matrix

def degree_matrix(adj_matrix: list[list[float]]) -> list[list[float]]:
    """Creates a degree matrix from an adjacency matrix.

    Args:
        adj_matrix (list[list[float]]): Adjacency matrix.

    Returns:
        list[list[float]]: Degree matrix.
    """
    deg_matrix = [[0 for _ in range(len(adj_matrix))] for _ in range(len(adj_matrix))]
    for i in range(len(adj_matrix)):
        num_connections = 0
        for j in range(len(adj_matrix[0])):
            if adj_matrix[i][j] != 0:
                num_connections += 1
        deg_matrix[i][i] = num_connections
    return deg_matrix

def graph_laplacian(degree_matrix: list[list[float]], adj_matrix: list[list[float]]) -> list[list[float]]:
    laplacian = degree_matrix
    for i in range(len(laplacian)):
        for j in range(len(laplacian)):
            laplacian[i][j] -= adj_matrix[i][j]
    return laplacian

def find_clusters(data: list[list[float]]) -> list[list[float]]:
    # First need to create adjacency matrix
    adj_matrix = adjacency_matrix(data)
    print(f'Found adjacency matrix:')
    print(adj_matrix)
    # Then find degree matrix
    deg_matrix = degree_matrix(adj_matrix)
    print(f'Found degree matrix:')
    print(deg_matrix)
    # Then graph laplacian (D - A)
    laplacian = graph_laplacian(deg_matrix, adj_matrix)
    print(f'Found graph laplacian:')
    print(laplacian)

    max_degree = max([deg_matrix[i][i] for i in range(len(deg_matrix))])
    # Then find eigenvalues/eigenvectors of graph laplacian
    eigen_pairs = eigen(laplacian, eigen_search_bounds=(0, 2 * max_degree), known_eigenvalues=[0.0], verbose=True)
    # print(f'eigen_pairs: {eigen_pairs}')
    evecs = [pair[1] for pair in eigen_pairs]
    # Now perform a change of basis for each data point
    # Construct vector P with eigenvectors as the columns
    # print(evecs)
    if len(evecs) < len(data):
        raise Exception('Unable to find all eigenvectors for data clusters.')
    # P = transverse(evecs)
    # print(f'Eigen basis')
    # print(np.array(P))
    
    plt.scatter(evecs[0], evecs[1])
    plt.show()

    # I am gonna try with two clusters first, so I will get the eigenvectors of the two smallest eigenvalues


    return

data = np.array(make_moons(30)[0])
plt.scatter(data.T[0], data.T[1])
plt.show()
print(data)
find_clusters(data)