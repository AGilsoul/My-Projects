try:
    from .NumMethods import Polynomial, Methods
except:
    from NumMethods import Polynomial, Methods

def build_identity_var(dim: int, var='x') -> list[list[Polynomial]]:
    """Builds an identity matrix consisting of polynomial types, multiplied by a given variable.

    Args:
        dim (int): Dimension of identity matrix to build.
        var (str, optional): Variable to multiply by. Defaults to 'x'.

    Returns:
        list[list[Polynomial]]: Identity matrix populated by polynomial types, mulitplied by var.
    """
    I = []
    # For every row
    for i in range(dim):
        # For every col
        cur_row = []
        for j in range(dim):
            if i == j:
                cur_row.append(Polynomial([0.0, 1.0], var=var))
            else:
                cur_row.append(Polynomial([0.0], var=var))
        I.append(cur_row)
    return I

def print_poly_matrix(p: list[list[Polynomial]]) -> None:
    """Prints polynomial matrices.

    Args:
        p (list[list[Polynomial]]): The polynomial matrix to print.
    """
    # For every row
    for i in range(len(p)):
        # For every col
        row_str = ''
        for j in range(len(p[i])):
            if j != len(p[i]) - 1:
                row_str += f'{str(p[i][j])}, '
            else:
                row_str += f'{str(p[i][j])}'
        print(row_str)
    return

def subtract_poly_matrices(A: list[list[Polynomial]], B: list[list[Polynomial]]) -> list[list[Polynomial]]:
    """Subtracts a polynomial matrix from another.

    Args:
        A (list[list[Polynomial]]): Matrix to subtract from.
        B (list[list[Polynomial]]): Matrix to subtract.

    Raises:
        Exception: Raised if matrices A and B are different sizes.

    Returns:
        list[list[Polynomial]]: Resultant matrix.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise Exception('Cannot subtract matrices with different dimensions.')
    
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i][j] = A[i][j] - B[i][j]
    return A

def convert_float_matrix_to_poly(A: list[list[float]], var: str) -> tuple[list[list[Polynomial]], list[list[int]]]:
    """Converts a float matrix to a polynomial matrix. Also returns a second matrix (zero mask) containing a 1 if the entry is non-zero, and a 0 if the entry is 0.
    We consider diagonal entries here to always be non-zero, as this was made with chacteristic determinants in mind.

    Args:
        A (list[list[float]]): The matrix to convert.
        var (str): The variable to use in the new polynomials.

    Returns:
        tuple[list[list[Polynomial]], list[list[int]]]: Result polynomial matrix, and a zero mask.
    """
    A_poly = []
    zero_mask = [[0.0 for _ in range(len(A[0]))] for _ in range(len(A))]
    # For every row
    for i in range(len(A)):
        cur_row = []
        cur_zero_row = []
        # For every col
        for j in range(len(A[i])):
            cur_row.append(Polynomial([A[i][j]], var=var))
            # won't be 0 cus A-λI
            if A[i][j] == 0 and i != j:
                continue

        A_poly.append(cur_row)
    return A_poly, zero_mask

# NEED TO IMPLEMENT BAREISS ALGORITHM
def poly_matrix_det(A: list[list[Polynomial]]) -> Polynomial:
    """Gets determinant of matrix of polynomials using Bareiss algorithm.

    Args:
        A (list[list[Polynomial]]): Matrix to calculate determinant of.

    Raises:
        Exception: Raised if the matrix passed is not a square matrix.

    Returns:
        Polynomial: Determinant of the matrix.
    """
    if len(A) != len(A[0]):
        raise Exception('sCannot find the determinant of a non-square matrix')

    M = []
    for i in range(len(A)):
        cur_row = []
        for j in range(len(A[i])):
            cur_row.append(Polynomial(A[i][j].coefs, A[i][j].var))
        M.append(cur_row)

    M_00 = Polynomial([1.0], var=M[0][0].var)
    n = len(A)
    sign = 1
    for k in range(n-1):
        for i in range(k+1, n):
            for j in range(k+1, n):
                if k != 0:
                    if M[k-1][k-1].is_zero():
                        old_above_row = M[k-1]
                        M[k-1] = M[i]
                        M[i] = old_above_row
                        sign *= -1
                    M[i][j] = Methods.poly_div((M[i][j] * M[k][k]) - (M[i][k] * M[k][j]), M[k-1][k-1])
                else:
                    M[i][j] = Methods.poly_div((M[i][j] * M[k][k]) - (M[i][k] * M[k][j]), M_00)

    return M[-1][-1] * sign

# def poly_matrix_det(A: list[list[Polynomial]], zero_mask: list[list[int]]=None) -> Polynomial:
#     """Gets determinant of matrix of polynomials.

#     Args:
#         A (list[list[Polynomial]]): Matrix to calculate determinant of.
#         zero_mask (list[list[int]], optional): Zero mask, not currently used, may be used for optimized determinant calculation. Defaults to None.

#     Raises:
#         Exception: Raised if the matrix passed is not a square matrix.

#     Returns:
#         Polynomial: Determinant of the matrix.
#     """
#     if len(A) != len(A[0]):
#         raise Exception('Cannot find the determinant of a non-square matrix')
#     # Set up determinant result
#     # Has to be a polynomial so we can do math with it
#     # Set the variable to be the same as those in the input matrix
#     det = Polynomial(0, var=A[0][0].var)

#     if zero_mask is None:
#         res = poly_matrix_det_rec_no_mask(A, (0, len(A)), (0, len(A)), [])
#     return res

# def poly_matrix_det_rec_no_mask(A: list[list[Polynomial]], row_bounds: tuple, col_bounds: tuple, excluded_columns: list[int]) -> Polynomial:
    # """Recursive helper for getting the determinant of a matrix of polynomials. This uses simple cofactor expansion. Called by poly_matrix_det.

    # Args:
    #     A (list[list[Polynomial]]): Matrix to find determinant of.
    #     row_bounds (tuple): Rows designating the current cofactor matrix.
    #     col_bounds (tuple): Columns designating the current cofactor matrix.
    #     excluded_columns (list[int]): Columns that are excluded, as they have been expaned upon already.

    # Returns:
    #     Polynomial: Determinant of the current cofactor matrix multiplied by the current cofactor.
    # """
    # # Set up determinant result
    # # Has to be a polynomial so we can do math with it
    # # Set the variable to be the same as those in the input matrix
    # det = Polynomial(0, var=A[0][0].var)

    # # If the sub matrix is 2x2, then we can just do ad - bc
    # if len(excluded_columns) == len(A) - 2:
    #     # First have to find the two columns we want
    #     cols = []
    #     for i in range(len(A[0])):
    #         if i not in excluded_columns:
    #             cols.append(i)
    #     det += A[row_bounds[0]][cols[0]] * A[row_bounds[1]-1][cols[1]]
    #     det -= A[row_bounds[0]][cols[1]] * A[row_bounds[1]-1][cols[0]]
    #     return det

    # else:
    #     # determine the sign of the first element
    #     row_parity = 1 if row_bounds[0] % 2 == 0 else -1
    #     col_parity = 1 if col_bounds[0] % 2 == 0 else -1
    #     starting_sign = row_parity * col_parity

    #     # So now we have the sign of each column
    #     # At this point I need to find a way to define the row and column bounds such that I can exclude 
    #     for i in range(col_bounds[0], col_bounds[1]):
    #         if i not in excluded_columns:
    #             if not A[0][i].is_zero():
    #                 if i == col_bounds[0]:
    #                     det += A[row_bounds[0]][i] * poly_matrix_det_rec_no_mask(A, (row_bounds[0]+1, row_bounds[1]), (col_bounds[0]+1, col_bounds[1]), excluded_columns + [i]) * starting_sign
    #                 elif i == col_bounds[1] - 1:
    #                     det += A[row_bounds[0]][i] * poly_matrix_det_rec_no_mask(A, (row_bounds[0]+1, row_bounds[1]), (col_bounds[0], col_bounds[1]-1), excluded_columns + [i]) * starting_sign
    #                 else:
    #                     det += A[row_bounds[0]][i] * poly_matrix_det_rec_no_mask(A, (row_bounds[0]+1, row_bounds[1]), (col_bounds[0], col_bounds[1]), excluded_columns + [i]) * starting_sign
    #         starting_sign *= -1
    #     return det


def main():
    M = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 16, 12],
        [13, 14, 15, 11]
    ]
    M_poly = convert_float_matrix_to_poly(M, 'x')[0]
    print(poly_matrix_det(M_poly))

if __name__ == '__main__':
    main()
