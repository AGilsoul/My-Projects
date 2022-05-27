import numpy as np
import math

axis_dict = {
    '': -1,
    'x': 0,
    'y': 1,
    'z': 2
}


class Polynomial:
    def __init__(self, coefs, powers, var='x'):
        self.var = var
        self.n = abs(max(powers))
        self.coefs = np.zeros(self.n + 1, np.float64)
        self.powers = np.arange(0, self.n + 1)
        for i in range(len(coefs)):
            self.coefs[powers[i]] = coefs[i]

    def __add__(self, b):
        new_poly = Polynomial(self.coefs, self.powers, self.var)
        if isinstance(b, int) or isinstance(b, float):
            new_poly.coefs[0] += b
            return new_poly

        if new_poly.powers[-1] >= b.powers[-1]:
            for i in range(len(b.powers)):
                new_poly.coefs[i] += b.coefs[i]
            new_poly.simplify()
        else:
            smaller_pows = new_poly.powers
            smaller_coefs = new_poly.coefs
            new_poly.powers = b.powers
            new_poly.coefs = b.coefs
            new_poly.n = b.n
            for i in range(len(smaller_pows)):
                new_poly.coefs[i] += smaller_coefs[i]
            new_poly.simplify()
        return new_poly

    def __sub__(self, b):
        return self + (b * -1)

    def __mul__(self, b):
        new_poly = Polynomial(self.coefs, self.powers, self.var)
        if isinstance(b, float) or isinstance(b, int):
            new_poly.coefs *= b

        elif isinstance(b, Polynomial):
            max_power = new_poly.powers[-1] + b.powers[-1]
            new_coefs = np.zeros(max_power + 1, np.float64)
            new_powers = np.arange(0, max_power + 1)
            for x in range(len(new_poly.coefs)):
                for y in range(len(b.coefs)):
                    cur_power = new_poly.powers[x] + b.powers[y]
                    new_coefs[cur_power] += new_poly.coefs[x] * b.coefs[y]
            new_poly.powers = new_powers
            new_poly.coefs = new_coefs
            new_poly.n = len(new_poly.coefs) - 1
            new_poly.simplify()
        return new_poly

    def __mod__(self, b):
        # copy polynomials
        poly_a = Polynomial(self.coefs, self.powers, self.var)
        poly_b = Polynomial(b.coefs, b.powers, b.var)

        if poly_b.powers[-1] > poly_a.powers[-1]:
            return poly_a

        degree_simplified_a = 0
        degree_simplified_b = 0
        if poly_a.coefs[0] == 0:
            degree_simplified_a = 1
            for i in range(1, len(poly_a.coefs)):
                if poly_a.coefs[i] == 0:
                    degree_simplified_a += 1
                else:
                    break
        if poly_b.coefs[0] == 0:
            degree_simplified_b = 1
            for i in range(1, len(poly_b.coefs)):
                if poly_b.coefs[i] == 0:
                    degree_simplified_b += 1
                else:
                    break

        min_degree = min(degree_simplified_a, degree_simplified_b)

        poly_a.change_order(poly_a.n - min_degree)
        poly_b.change_order(poly_b.n - min_degree)

        a_index_count = 0
        quotient = Polynomial([0], [0])
        large_divisor = Polynomial([poly_b.coefs[-1]], [poly_b.powers[-1]])
        while poly_a.coefs[-1] != 0 and a_index_count < len(poly_b.coefs):
            large_dividend = Polynomial([poly_a.coefs[-1]], [poly_a.powers[-1]])
            q = large_dividend.single_term_div(large_divisor)
            poly_a -= (q * poly_b)
            quotient += q
            poly_a.simplify()
            a_index_count += 1

        poly_a.simplify()
        return poly_a

    def single_term_div(self, b):
        poly_a = Polynomial(self.coefs, self.powers, self.var)
        poly_b = Polynomial(b.coefs, b.powers, b.var)

        a_index = -1
        for i in range(len(poly_a.coefs)):
            if poly_a.coefs[i] != 0:
                a_index = i

        b_index = -1
        for i in range(len(poly_b.coefs)):
            if poly_b.coefs[i] != 0:
                b_index = i

        if a_index == -1 or b_index == -1:
            return Polynomial([0], [0])

        #print(str(poly_a.coefs[a_index]) + "x^" + str(poly_a.powers[a_index]) + " / " + str(poly_b.coefs[b_index]) + "x^" + str(poly_b.powers[b_index]))
        new_coef = poly_a.coefs[a_index] / poly_b.coefs[b_index]
        new_power = poly_a.powers[a_index] - poly_b.powers[b_index]
        res = Polynomial([new_coef], [new_power])
        res.simplify()
        #print(res)
        #print()
        return res

    def __truediv__(self, b):
        # copy polynomials
        poly_a = Polynomial(self.coefs, self.powers, self.var)
        poly_b = Polynomial(b.coefs, b.powers, b.var)

        if poly_b.powers[-1] > poly_a.powers[-1]:
            return Polynomial([0], [0], self.vr)

        degree_simplified_a = 0
        degree_simplified_b = 0
        if poly_a.coefs[0] == 0:
            degree_simplified_a = 1
            for i in range(1, len(poly_a.coefs)):
                if poly_a.coefs[i] == 0:
                    degree_simplified_a += 1
                else:
                    break
        if poly_b.coefs[0] == 0:
            degree_simplified_b = 1
            for i in range(1, len(poly_b.coefs)):
                if poly_b.coefs[i] == 0:
                    degree_simplified_b += 1
                else:
                    break

        min_degree = min(degree_simplified_a, degree_simplified_b)

        poly_a.change_order(poly_a.n - min_degree)
        poly_b.change_order(poly_b.n - min_degree)

        a_index_count = 0
        quotient = Polynomial([0], [0], self.var)
        large_divisor = Polynomial([poly_b.coefs[-1]], [poly_b.powers[-1]], poly_b.var)
        while poly_a.coefs[-1] != 0 and a_index_count < len(poly_b.coefs):
            large_dividend = Polynomial([poly_a.coefs[-1]], [poly_a.powers[-1]], poly_a.var)
            q = large_dividend.single_term_div(large_divisor)
            poly_a -= (q * poly_b)
            quotient += q
            poly_a.simplify()
            a_index_count += 1

        quotient.simplify()
        return quotient

    def simplify(self):
        num_coefs = 0
        for i in reversed(range(len(self.coefs))):
            if self.coefs[i] == 0:
                num_coefs += 1
            else:
                break
        if num_coefs != self.n + 1:
            for i in range(num_coefs):
                self.coefs = np.delete(self.coefs, -1)
                self.powers = np.delete(self.powers, -1)
            self.n = len(self.powers) - 1

    def change_order(self, new_order):
        self.coefs = self.coefs[self.n - new_order:]
        self.powers = np.arange(new_order + 1)
        self.n = new_order

    def eval(self, x):
        x = np.float64(x)
        res = sum(self.coefs[i] * x**self.powers[i] for i in range(len(self.powers)))
        return res

    def __lt__(self, b):
        if self.powers[-1] < b.powers[-1]:
            return True
        elif self.powers[-1] > b.powers[-1]:
            return False
        else:
            inequality = False
            coef_count = 0
            while not inequality and coef_count < len(self.coefs):
                index = len(self.coefs) - coef_count - 1
                if self.coefs[index] < b.coefs[index]:
                    return True
                elif self.coefs[index] > b.coefs[index]:
                    return False
                else:
                    coef_count += 1
            return False

    def __le__(self, b):
        if self < b:
            return True
        return self == b

    def __gt__(self, b):
        if self.powers[-1] > b.powers[-1]:
            return True
        elif self.powers[-1] < b.powers[-1]:
            return False
        else:
            inequality = False
            coef_count = 0
            while not inequality and coef_count < len(self.coefs):
                index = len(self.coefs) - coef_count - 1
                if self.coefs[index] > b.coefs[index]:
                    return True
                elif self.coefs[index] < b.coefs[index]:
                    return False
                else:
                    coef_count += 1
            return False

    def __ge__(self, b):
        if self > b:
            return True
        else:
            return self == b

    def __eq__(self, b):
        if len(self.coefs) != len(b.coefs):
            return False
        for i in range(len(self.coefs)):
            if self.coefs[i] != b.coefs[i] or self.powers[i] != b.powers[i]:
                return False
        return True

    def __ne__(self, b):
        return not self == b

    def __str__(self):
        out = ""
        first_term = True
        for i in range(len(self.powers)):
            if self.coefs[i] != 0:
                if first_term and self.coefs[i] < 0:
                    out += "-"

                if abs(self.coefs[i]) != 1 or (first_term and abs(self.coefs[i]) == 1):
                    out += str(abs(self.coefs[i]))

                if self.powers[i] == 1:
                    out += self.var
                elif self.powers[i] != 0:
                    out += self.var + "^" + str(self.powers[i])

                if i != len(self.powers) - 1:
                    if self.coefs[i + 1] >= 0:
                        out += " + "
                    elif self.coefs[i + 1] < 0:
                        out += " - "
                first_term = False
        return out


class Rational:
    def __init__(self, p, q):
        if isinstance(p, Polynomial):
            self.numerator = p
        else:
            self.numerator = Polynomial([p], [0])

        if isinstance(q, Polynomial):
            self.denominator = q
        else:
            self.denominator = Polynomial([q], [0])

    def __add__(self, b):
        new_numerator = Polynomial(self.numerator.coefs, self.numerator.powers, self.numerator.var)
        new_denominator = Polynomial(self.denominator.coefs, self.denominator.powers, self.denominator.var)
        new_rational = Rational(new_numerator, new_denominator)
        if isinstance(b, Rational):
            if self.denominator != b.denominator:
                new_rational.numerator *= b.denominator
                new_rational.denominator *= b.denominator
                added_poly = b.numerator * new_denominator
                new_rational.numerator += added_poly
            else:
                new_rational.numerator += b.numerator
        else:
            new_rational.numerator += b * new_rational.denominator

        new_rational.simplify()
        return new_rational

    def __sub__(self, b):
        return self + (b * -1)

    def __mul__(self, b):
        new_numerator = Polynomial(self.numerator.coefs, self.numerator.powers, self.numerator.var)
        new_denominator = Polynomial(self.denominator.coefs, self.denominator.powers, self.denominator.var)
        new_rational = Rational(new_numerator, new_denominator)
        if isinstance(b, Rational):
            new_rational.numerator *= b.numerator
            new_rational.denominator *= b.denominator
        else:
            new_rational.numerator *= b
        new_rational.simplify()
        return new_rational

    def __truediv__(self, b):
        new_num = self.numerator
        new_den = self.denominator
        new_poly = Rational(new_num, new_den)
        new_poly.denominator *= b.numerator
        new_poly.numerator *= b.denominator
        new_poly.simplify()
        return new_poly

    def eval(self, x):
        return self.numerator.eval(x) / self.denominator.eval(x)

    def definite_simplify(self):
        self.numerator /= self.denominator
        self.denominator = Polynomial([1], [0])

    def simplify(self):
        coefs_zeros = True
        for i in range(len(self.numerator.coefs)):
            if self.numerator.coefs[i] != 0:
                coefs_zeros = False
                break
        if coefs_zeros:
            self.denominator = Polynomial([1], [0])



        degree_simplified_a = 0
        degree_simplified_b = 0
        if self.numerator.coefs[0] == 0:
            degree_simplified_a = 1
            for i in range(1, len(self.numerator.coefs)):
                if self.numerator.coefs[i] == 0:
                    degree_simplified_a += 1
                else:
                    break
        if self.denominator.coefs[0] == 0:
            degree_simplified_b = 1
            for i in range(1, len(self.denominator.coefs)):
                if self.denominator.coefs[i] == 0:
                    degree_simplified_b += 1
                else:
                    break

        min_degree = min(degree_simplified_a, degree_simplified_b)

        self.numerator.change_order(self.numerator.n - min_degree)
        self.denominator.change_order(self.denominator.n - min_degree)

    def __str__(self):
        coefs_zero = True
        for i in range(len(self.numerator.coefs)):
            if self.numerator.coefs[i] != 0:
                coefs_zero = False
                break
        if coefs_zero:
            return "0"
        if len(self.denominator.coefs) == 1 and self.denominator.coefs[0] == 1:
            return str(self.numerator)
        out = "(" + str(self.numerator) + ") / (" + str(self.denominator) + ")"
        return out



def vector_magnitude(a):
    return math.sqrt(np.dot(a, a.T))


def vector_stretch(a, factor, axis=''):
    n = a.shape[0]
    axis_index = axis_dict.get(axis)
    stretch_matrix = np.zeros([n, n], np.float64)
    for i in range(n):
        if i == axis_index or axis_index == -1:
            stretch_matrix[i][i] = factor
        else:
            stretch_matrix[i][i] = 1
    return np.dot(stretch_matrix, a)


def vector_rotate(a, radians):
    rotation_matrix = np.array(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)]
        ]
    )
    return np.dot(rotation_matrix, a)


def vector_rotate_3d(a, radians, axis='x'):
    if axis == 'x':
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, math.cos(radians), -math.sin(radians)],
                [0, math.sin(radians), math.cos(radians)]
            ]
        )
        return np.dot(rotation_matrix, a)
    elif axis == 'y':
        rotation_matrix = np.array(
            [
                [math.cos(radians), 0, math.sin(radians)],
                [0, 1, 0],
                [-math.sin(radians), 0, math.cos(radians)]
            ]
        )
        return np.dot(rotation_matrix, a)
    else:
        rotation_matrix = np.array(
            [
                [math.cos(radians), -math.sin(radians), 0],
                [math.sin(radians), math.cos(radians), 0],
                [0, 0, 1]
            ]
        )
        return np.dot(rotation_matrix, a)


def gcd(a, b):
    poly_a = a
    poly_b = b
    if poly_a < poly_b:
        temp = poly_a
        poly_a = poly_b
        poly_b = temp
    zero_poly = Polynomial([0], [0])
    while poly_b > zero_poly:
        # fix modulus division for polynomials
        temp = poly_a % poly_b
        poly_a = poly_b
        poly_b = temp
    return poly_a


def gaussian_elimination(a):
    n = a.shape[0]
    det_factors = []
    for x in range(n - 1):
        for i in range(x + 1, n):
            # current row xth term divided by above row's xth term
            factor = a[i][x] / a[x][x]
            a[i] -= a[x] * factor
    return a, det_factors


def reduce_matrix(a):
    return gaussian_elimination(a)


def rational_identity(n):
    id_matrix = np.empty([n, n], Rational)
    for x in range(n):
        for y in range(n):
            if x == y:
                id_matrix[x][y] = Rational(Polynomial([1], [1], 'λ'), Polynomial([1], [0], 'λ'))
            else:
                id_matrix[x][y] = Rational(Polynomial([0], [0], 'λ'), Polynomial([1], [0], 'λ'))
    return id_matrix


def matrix_eigenvalues(a):
    return np.linalg.eig(a)[0]


def matrix_eigenvectors(a):
    return np.linalg.eig(a)[1].T


def matrix_get_column(a, col_index):
    return a.T[col_index]


def matrix_trace(a):
    return sum([a[i][i] for i in range(a.shape[0])])


def matrix_diagonal_prod(a):
    factor = Polynomial([1], [0], a[0][0].numerator.var)
    product = Rational(factor, factor)
    n = a.shape[0]
    for i in range(n):
        product *= a[i][i]
    return product


def quick_determinant(a):
    reduced, factors = reduce_matrix(a)
    res = matrix_diagonal_prod(reduced)
    for i in range(len(factors)):
        res *= factors[i]
    return res


def matrix_to_rational(a, var='x'):
    n = a.shape[0]
    new_matrix = np.empty([n, n], Rational)
    for x in range(n):
        for y in range(n):
            new_matrix[x][y] = Rational(Polynomial([a[x][y]], [0], var), Polynomial([1], [0], var))
    return new_matrix


def print_matrix(a):
    out = ""
    for x in range(len(a)):
        out += "["
        for y in range(len(a)):
            out += str(a[x][y])
            if y != len(a) - 1:
                out += ", "
            else:
                out += "]\n"
    print(out)


def char_polynomial(a):
    identity_matrix = rational_identity(a.shape[0])
    v_mat = matrix_to_rational(a, 'λ')
    pre_eigen = v_mat - identity_matrix
    det = quick_determinant(pre_eigen)
    det.definite_simplify()
    return det


mat = np.array(
    [
        [1, 4, 5, 2, 3],
        [4, 6, 6, 3, 5],
        [5, 6, 3, 3, 2],
        [2, 3, 3, 9, 2],
        [3, 5, 2, 2, 4]
    ]
)
print(mat)
char_p = char_polynomial(mat)
print(char_p)

