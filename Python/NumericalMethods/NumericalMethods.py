import random
import matplotlib.pyplot as plt


class ComplexNumber:
    def __init__(self, Re, Im=0.0):
        self.Re = Re
        self.Im = Im

    def modulus(self):
        return (self.Re**2 + self.Im**2)**(1/2)

    def __add__(self, other):
        new_number = ComplexNumber(self.Re, self.Im)
        if type(new_number) == type(other):
            new_number.Re += other.Re
            new_number.Im += other.Im
        else:
            new_number.Re += other
        return new_number

    def __iadd__(self, other):
        res = self + other
        self.Re = res.Re
        self.Im = res.Im
        return self

    def __mul__(self, other):
        new_number = ComplexNumber(self.Re, self.Im)
        if type(new_number) == type(other):
            new_number.Re = (new_number.Re * other.Re) - (new_number.Im * other.Im)
            new_number.Im = (new_number.Re * other.Im) + (new_number.Im * other.Re)
        else:
            new_number.Re *= other
            new_number.Im *= other
        return new_number

    def __imul__(self, other):
        res = self * other
        self.Re = res.Re
        self.Im = res.Im
        return self

    def __sub__(self, other):
        res = (self * -1) + other
        return res

    def __isub__(self, other):
        res = self - other
        self.Re = res.Re
        self.Im = res.Im
        return res

    def __str__(self):
        if self.Im < 0:
            return f'{self.Re} - {abs(self.Im)}i'
        return f'{self.Re} + {self.Im}i'




class Function:
    def __init__(self, var):
        self.var = var
        return

    def __call__(self, x):
        return x


class FactoredPolynomial(Function):
    def __init__(self, roots, var):
        super().__init__(var)
        self.roots = roots

    def __call__(self, x):
        val = 1
        for root in self.roots:
            val *= (x - root)
        return val



class Polynomial(Function):
    def __init__(self, coefs, var='x'):
        super().__init__(var)
        self.coefs = coefs

    def __call__(self, x):
        return sum([self.coefs[i] * x ** i for i in range(len(self.coefs))])

    def shorten(self):
        for i in reversed(range(len(self.coefs))):
            if self.coefs[i] == 0:
                self.coefs = self.coefs.pop()
                break

    def __add__(self, p):
        if type(self) == type(p):
            new_coefs = [0 for _ in range(max(len(p.coefs), len(self.coefs)))]
            for i in range(len(self.coefs)): new_coefs[i] += self.coefs[i]
            for i in range(len(p.coefs)): new_coefs[i] += p.coefs[i]
        else:
            new_coefs = self.coefs
            new_coefs[0] += p
        self.shorten()
        return Polynomial(new_coefs)

    def __iadd__(self, p):
        if type(self) == type(p):
            new_coefs = [0 for _ in range(max(len(p.coefs), len(self.coefs)))]
            for i in range(len(self.coefs)): new_coefs[i] += self.coefs[i]
            for i in range(len(p.coefs)): new_coefs[i] += p.coefs[i]
            self.coefs = new_coefs
        else:
            self.coefs[0] += p
        self.shorten()
        return self

    def __mul__(self, p):
        if type(self) == type(p):
            new_coefs = [0 for _ in range((len(self.coefs)) + (len(p.coefs)) - 1)]
            for i in range(len(self.coefs)):
                for j in range(len(p.coefs)):
                    cur_power = i + j
                    new_coefs[cur_power] += self.coefs[i] * p.coefs[j]
        else:
            new_coefs = self.coefs
            for i in range(len(self.coefs)):
                new_coefs[i] *= p
        self.shorten()
        return Polynomial(new_coefs)

    def __imul__(self, p):
        if type(self) == type(p):
            new_coefs = [0 for _ in range((len(self.coefs)) + (len(p.coefs)) - 1)]
            for i in range(len(self.coefs)):
                for j in range(len(p.coefs)):
                    cur_power = i + j
                    new_coefs[cur_power] += self.coefs[i] * p.coefs[j]
            self.coefs = new_coefs
        else:
            for i in range(len(self.coefs)):
                self.coefs[i] *= p
        self.shorten()
        return self

    def __sub__(self, p):
        return self + -1 * p

    def __isub__(self, p):
        self.coefs = (self - p).coefs
        self.shorten()
        return self

    def __str__(self):
        first_non_zero = 0
        for i in range(len(self.coefs)):
            if self.coefs[i] != 0:
                first_non_zero = i
                break
        output = f'{self.coefs[first_non_zero]}'
        if first_non_zero != 0:
            output += f'{self.var}^{first_non_zero}'
        for i in range(first_non_zero + 1, len(self.coefs)):
            abs_val = abs(self.coefs[i])
            if self.coefs[i] < 0:
                output += f' - {abs_val}{self.var}^{i}'
            elif self.coefs[i] > 0:
                output += f' + {abs_val}{self.var}^{i}'
        return output


class Methods:
    @staticmethod
    def approx_deriv(func, x, dx=0.001):
        x1 = x - dx/2
        x2 = x + dx/2
        dy = func(x2) - func(x1)
        return dy / dx

    @staticmethod
    def linear_zero(m, p):
        if m == 0: return p[0]
        x = p[0]
        y = p[1]
        return x - y/m

    @staticmethod
    def newton_approx(func, num_zeros=1, x_range=(-1, 1), num_iterations=100, max_attempts=100):
        zeros = []
        range_min = min(x_range)
        range_max = max(x_range)
        for zero in range(num_zeros):
            num_attempts = 0
            while True:
                x = random.uniform(range_min, range_max)
                for i in range(num_iterations):
                    deriv = Methods.approx_deriv(func, x)
                    x = round(Methods.linear_zero(deriv, (x, func(x))), 5)
                num_attempts += 1
                if x not in zeros:
                    print('found zero')
                    zeros.append(x)
                    break
                if num_attempts >= max_attempts:
                    print('exceeded attempts')
                    break
        return sorted(zeros)


poly = Polynomial([0, 2, 5, 3])
val = -7
print(f'polynomial {poly} at x={val}: {poly(val)}')
print(f'approximate zeros of polynomial {poly}: x={Methods.newton_approx(poly, num_zeros=3, x_range=(-1, 1))}')\

p1 = Polynomial([1, -2, 3, -13, 2, 4])
p2 = poly * p1


print(f'multiplying polynomial {poly} by {p1}: {p2}')
zeros = Methods.newton_approx(p2, num_zeros=8, x_range=(-10, 10))
print(f'approximate zeros of polynomial {p2}: {zeros}')

