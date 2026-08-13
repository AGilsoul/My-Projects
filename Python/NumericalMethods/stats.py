import numpy as np
from eigen import eigen
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def sample_mean(x: list[float]) -> float:
    """Calculate the sample mean of data samples from one random variable.

    Args:
        x (list[float]): Random variable X data sample.

    Returns:
        float: Sample mean of X.
    """
    return sum(x) / len(x)

def variance(x: list[float], E_x: float=None) -> float:
    """Calculates the sample variance of data samples from one random variable.

    Args:
        x (list[float]): Random variable X data sample.
        E_x (float, optional): Sample mean of X. Defaults to None.

    Returns:
        float: Sample variance of X.
    """
    return covariance(x, x, E_x, E_x)

def covariance(x: list[float], y: list[float], E_x: float=None, E_y: float=None) -> float:
    """Calculates sample covariance of data samples from two random variables.

    Args:
        x (list[float]): First random variable X data sample.
        y (list[float]): Second random variable Y data sample.
        E_x (float, optional): Sample mean of X. Defaults to None.
        E_y (float, optional): Sample mean of Y. Defaults to None.

    Raises:
        Exception: Raised if samples have different lengths.

    Returns:
        float: Sample covariance of X and Y.
    """
    n = len(x)
    if n != len(y):
        raise Exception('Data must have the same length.')

    if E_x is None:
        E_x = sum(x) / n
    if E_y is None:
        E_y = sum(y) / n

    return sum([(x[i] - E_x) * (y[i] - E_y) for i in range(n)]) / (n-1)

def covariance_matrix(data: list[list[float]]) -> list[list[float]]:
    """Calculates the covariance matrix for a data sample.

    Args:
        data (list[list[float]]): Date to find the covariance matrix of (variable x data points).

    Returns:
        list[list[float]]: The covariance matrix of the data sample.
    """
    # Get number of random variables
    dim = len(data)

    expected_values = [sample_mean(data[i]) for i in range(dim)]

    cov_mat = [[0 for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for j in range(i, dim):
            cov_ij = covariance(data[i], data[j], expected_values[i], expected_values[j])
            cov_mat[i][j] = cov_ij
            cov_mat[j][i] = cov_ij

    return cov_mat

def pca(data: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Finds the principle components of a dataset.

    Args:
        data (list[list[float]]): The data to analyze (variable x data points).

    Returns:
        tuple[list[float], list[list[float]]]: Returns the variance along each principle component axis (eigenvalues) and each principle compnent axis (eigenvectors).
    """
    cov_mat = covariance_matrix(data)
    return eigen(cov_mat)

def make_blob_dataset():
    X, y = make_blobs(n_samples=10000, centers=1, n_features=2,
                random_state=0)

    X = X.T
    X[1] /= 5
    X = X.T

    rot_matrix = np.array([
        [np.cos(np.pi/2), -np.sin(np.pi/2)],
        [np.sin(np.pi/2), np.cos(np.pi/2)]
    ])

    for i in range(len(X)):
        X[i] = rot_matrix @ X[i]
    X = X.T
    return X

def main():
    X = make_blob_dataset()

    x_mean = sample_mean(X[0])
    y_mean = sample_mean(X[1])

    eigenvalues, eigenvectors = pca(X)
    eigenvectors = np.array(eigenvectors)
    for i in range(len(eigenvectors)):
        eigenvectors[i] = eigenvectors[i] / np.linalg.norm(eigenvectors[i]) * eigenvalues[i]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0], X[1])
    e_origins = np.array([[x_mean, x_mean], [y_mean, y_mean]])
    ax.quiver(*e_origins, eigenvectors[:,0], eigenvectors[:,1], color=['lightcoral', 'lightsteelblue'], scale=5)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.show()

if __name__ == '__main__':
    main()
