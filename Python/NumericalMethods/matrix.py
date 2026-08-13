import numpy as np

def solve_system(A_mat: list[list[float]]) -> list[float]:
    """Takes in matrix A for Av = 0, solves for vector v.

    Args:
        A_mat (list[list[float]]): Matrix representing system to solve.

    Returns:
        list[float]: Solution vector v.
    """

    A = []
    for i in range(len(A_mat)):
        cur_row = []
        for j in range(len(A_mat[i])):
            cur_row.append(A_mat[i][j])
        A.append(cur_row)

    # First, convert to upper triangular matrix
    # So for every column i, we will make sure that row i is the only row with an entry in that column, except for rows above it
    # Don't need to do this for the last row, it has no rows beneath it
    for i in range(len(A) - 1):
        # If column i does not have an entry at row i, fix it
        if A[i][i] == 0:
            # Find another row (beneath the current row) with an entry at column i
            # Because of the order, the rows beneath should not have any entries in columns before column i
            row_with_val = -1
            for j in range(i+1, len(A)):
                if A[j][i] != 0:
                    row_with_val = j
                    break
            if row_with_val == -1:
                continue
            # Now add the row with a value in said column, to the current row
            for j in range(i, len(A)):
                A[i][j] += A[row_with_val][j]
        # Now zero out the entries in column i for every row beneath this one
        for j in range(i+1, len(A)):
            factor = A[j][i] / A[i][i] 
            for k in range(i, len(A)):
                A[j][k] -= factor * A[i][k]

        # Now in the end, we want to set every diagonal entry to 1 to make this easier to think about for me
        diag_entry = A[i][i]
        for j in range(i, len(A)):
            A[i][j] /= diag_entry

    # Now go through and make any entries really close to 0, 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if abs(A[i][j]) < 1e-5:
                A[i][j] = 0

    # Now for each row, we should have some form of:
    # C1*var1 + C2*var2 + ... + CN*varN = 0
    # The bottommost row should be something like 
    # 0*var1 + ... + 0*varN = 0
    # OR
    # 0*var1 + ... + 1*varN = 0
    # The second from the bottom row should be something like
    # 0*var1 + ... + 1*var(N-1) + CN*varN = 0
    # So solve for each variable with this in mind
    v = [None for _ in range(len(A))]
    # Find solution from each row if possible
    for i in reversed(range(len(A))):
        if A[i][i] != 0:
            v[i] = - sum([v[j] * A[i][j] for j in range(i+1, len(A))])
        else:
            v[i] = 1

    return v

def build_identity(n: int) -> list[list[float]]:
    """Builds an nxn identity matrix.

    Args:
        n (int): Dimension of the identiy matrix.

    Returns:
        list[list[float]]: nxn identity matrix.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def transverse(A: list[list]) -> list[list]:
    """Builds transverse of matrix A.

    Args:
        A (list[list]): Input matrix.

    Returns:
        list[list]: A^T.
    """
    # Just switch A[i][j] with A[j][i]
    T = []
    for i in range(len(A)):
        cur_row = []
        for j in range(len(A)):
            cur_row.append(A[j][i])
        T.append(cur_row)
    return T

def matrix_inverse(A: list[list]) -> list[list]:
    """Computes the inverse (if possible) of matrix A.

    Args:
        A (list[list]): The matrix to invert.

    Returns:
        list[list]: A^-1.
    """
    # Have to create the identity matrix to manipulate alongside the input matrix A
    I = build_identity(len(A))
    # First, convert to upper triangular matrix
    # So for every column i, we will make sure that row i is the only row with an entry in that column, except for rows above it
    # Don't need to do this for the last row, it has no rows beneath it
    for i in range(len(A) - 1):
        # If column i does not have an entry at row i, fix it
        if A[i][i] == 0:
            # Find another row (beneath the current row) with an entry at column i
            # Because of the order, the rows beneath should not have any entries in columns before column i
            row_with_val = -1
            for j in range(i+1, len(A)):
                if A[j][i] != 0:
                    row_with_val = j
                    break
            if row_with_val == -1:
                continue
            # Now add the row with a value in said column, to the current row
            for j in range(i, len(A)):
                A[i][j] += A[row_with_val][j]
                I[i][j] += I[row_with_val][j]
        # Now zero out the entries in column i for every row beneath this one
        for j in range(i+1, len(A)):
            factor = A[j][i] / A[i][i] 
            for k in range(len(A)):
                A[j][k] -= factor * A[i][k]
                I[j][k] -= factor * I[i][k]

    # Now go through and make any entries really close to 0, 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if abs(A[i][j]) < 1e-5:
                A[i][j] = 0

    # Now we need to go from upper triangular to identity! Same as above kinda, but backwards and up
    # Don't have to do this for the top row, because no rows above!
    for i in reversed(range(1, len(A))):
        # If column i does not have an entry at row i, skip it (shouldn't happen for graph laplacian)
        if A[i][i] == 0:
            continue
        # Now zero out the entries in column i for every row above this one
        for j in reversed(range(i)):
            factor = A[j][i] / A[i][i] 
            for k in range(len(A)):
                A[j][k] -= factor * A[i][k]
                I[j][k] -= factor * I[i][k]

    # Now go through and make any entries really close to 0, 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if abs(A[i][j]) < 1e-5:
                A[i][j] = 0

    # Now in the end, we want to set every diagonal entry to 1 to make this easier to think about for me
    for i in range(len(A)):
        diag_entry = A[i][i]
        if diag_entry == 0:
            continue
        for j in range(len(A)):
            A[i][j] /= diag_entry
            I[i][j] /= diag_entry

    return I