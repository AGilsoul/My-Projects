import numpy as np
try:
    from .poly_matrix import *
    from .matrix import *
    from .NumMethods import Polynomial, Methods
except:
    from poly_matrix import *
    from matrix import *
    from NumMethods import Polynomial, Methods
import matplotlib.pyplot as plt

LAMBDA = 'λ'

def eigen(A: list[list[float]], eigen_search_bounds=(0, 20), known_eigenvalues=None, verbose=False) -> tuple[list[float], list[list[float]]]:
    """Calculates the eigenvalues and eigenvectors of a given matrix.

    Args:
        A (list[list[float]]): Matrix to find eigenvalues/eigenvectors of.
        eigen_search_bounds (tuple, optional): Search bounds used in root finding to approximate eigenvalues. Defaults to (-20, 20).
        verbose (bool): If true, will print more output. Defaults to False.

    Returns:
        tuple[list[float], list[list[float]]]: Tuple containing the eigenvalues and eigenvectors of the given matrix A.
    """
    # Convert the matrix to contain Polynomial types
    A_poly, zero_mask = convert_float_matrix_to_poly(A, var=LAMBDA)

    if verbose:
        print(f'Matrix as polynomial:')
        print_poly_matrix(A_poly)

    # Get the corresponding identity matrix multiplied by λ
    lam_id = build_identity_var(len(A_poly), var=LAMBDA)

    # Get A-λI
    A_min_lam_id = subtract_poly_matrices(A_poly, lam_id)

    if verbose:
        print(f'A-λI:')
        print_poly_matrix(A_min_lam_id)
    
    # Find the characteristic polynomial by taking the determinant of A-λI
    # char_eqn = poly_matrix_det(A_min_lam_id, zero_mask)
    char_eqn = poly_matrix_det(A_min_lam_id)

    if verbose:
        print(f'Characteristic polynomial: {char_eqn}')
    
    # Approximate the solutions to chararacteristic polynomial to find eigenvalues
    # Number of total (including nonunique) solutions should be number of diagonal entries in A
    eigenvalues = Methods.newton_poly_iterative(char_eqn, len(A), known_zeros=known_eigenvalues, x_range=eigen_search_bounds, num_iterations=int(1e6), max_attempts=1000)

    if verbose:
        print(f'Found eigenvalues:')
        print(eigenvalues)

    # Now calculate actual A-λI so we can solve (A-λI)v = 0
    eigenvectors = []
    for lam in eigenvalues:
        # print(f'On eigenvalue {lam}')
        # print(f'Plugging into')
        # print_poly_matrix(A_min_lam_id)
        # Now calculate actual A-λI so we can solve (A-λI)v = 0
        A_λI = []
        for i in range(len(A_min_lam_id)):
            cur_row = []
            for j in range(len(A_min_lam_id)):
                cur_row.append(A_min_lam_id[i][j](lam))
            A_λI.append(cur_row)

        # Now solve (A-λI)v = 0
        v = solve_system(A_λI)
        eigenvectors.append((lam, v))

    if verbose:
        print(f'Found eigenvectors:')
        print(eigenvectors)

    return eigenvectors

# Generates a unit circle of n^dim vectors in dim dimensions
def unit_circle_vectors(n, dim):
    vectors = []
    if dim == 2:
        for i in range(n):
            vectors.append([np.sin(2 * i * np.pi / n), np.cos(2 * i * np.pi / n)])
    elif dim == 3:
        for theta in range(n):
            for phi in range(n):
                vectors.append([np.sin(2*phi*np.pi/n)*np.cos(2*theta*np.pi/n), np.sin(2*phi*np.pi/n)*np.sin(2*theta*np.pi/n), np.cos(2*phi*np.pi/n)])
    return np.array(vectors)

# Plots a matrix transformation along with the eigenvectors
def plot_transformation(matrix, n, lims=(-10,10)):
    np_matrix = np.array(matrix)

    eigen_pairs = eigen(matrix, verbose=True)
    ev = [pair[0] for pair in eigen_pairs]
    eigenvectors = [pair[1] for pair in eigen_pairs]

    print(f'Matrix:')
    print(np.array(matrix))
    print(f'Eigenvalues: {ev}')
    print(F'Eigenvectors: {eigenvectors}')

    unscaled_evecs = np.array((eigenvectors))
    evecs = []
    for i in range(len(unscaled_evecs)):
        evecs.append(np_matrix @ unscaled_evecs[i])
    evecs = np.array(evecs)

    # This creates a bunch of unit circle vectors to transform
    unit_vectors = unit_circle_vectors(n, len(matrix))
    transformed_unit_vectors = []
    for i in range(len(unit_vectors)):
        transformed_unit_vectors.append(np_matrix @ unit_vectors[i])
    transformed_unit_vectors = np.array(transformed_unit_vectors)

    if len(matrix) == 2:
        # Plotting stuff
        fig = plt.figure()
        ax = fig.add_subplot(111)

        origins = np.array([[0 for _ in range(len(unit_vectors))], [0 for _ in range(len(unit_vectors))]])
        ax.quiver(*origins, unit_vectors[:,0], unit_vectors[:,1], color=['g' for _ in range(len(unit_vectors))], scale=5)
        t_origins = np.array([[0 for _ in range(len(transformed_unit_vectors))], [0 for _ in range(len(transformed_unit_vectors))]])
        ax.quiver(*origins, transformed_unit_vectors[:,0], transformed_unit_vectors[:,1], color=['greenyellow' for _ in range(len(transformed_unit_vectors))], scale=5)

        e_origins = np.array([[0, 0], [0, 0]])
        ax.quiver(*e_origins, unscaled_evecs[:,0], unscaled_evecs[:,1], color=['r', 'b'], scale=5)
        ax.quiver(*e_origins, evecs[:,0], evecs[:,1], color=['lightcoral', 'lightsteelblue'], scale=5)

        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)

        fig.tight_layout()
        plt.show()
    elif len(matrix) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        unit_vector_origins = np.array([[0 for _ in range(len(unit_vectors))], [0 for _ in range(len(unit_vectors))], [0 for _ in range(len(unit_vectors))]])
        unit_vectors = unit_vectors.T
        ax.quiver(*unit_vector_origins, unit_vectors[0], unit_vectors[1], unit_vectors[2], color=['g' for _ in range(len(unit_vectors[0]))])

        transformed_unit_vectors = transformed_unit_vectors.T
        ax.quiver(*unit_vector_origins, transformed_unit_vectors[0], transformed_unit_vectors[1], transformed_unit_vectors[2], color=['greenyellow' for _ in range(len(transformed_unit_vectors[0]))])

        unscaled_evecs = unscaled_evecs.T
        evecs = evecs.T
        e_origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ax.quiver(*e_origins, unscaled_evecs[0], unscaled_evecs[1], unscaled_evecs[2], color=['r', 'b'])
        ax.quiver(*e_origins, evecs[0], evecs[1], evecs[2], color=['lightcoral', 'lightsteelblue'])

        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
        ax.set_zlim(lims[0], lims[1])

        fig.tight_layout()
        plt.show()

def main():
    matrix = [
        [1, 2],
        [0, -1]
    ]

    plot_transformation(matrix, 100)

if __name__ == '__main__':
    main()