import random


class Function:
    def __init__(self):
        return

    def __call__(self, x):
        return x


class Polynomial(Function):
    def __init__(self, coefs):
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
        return Polynomial(new_coefs)

    def __iadd__(self, p):
        if type(self) == type(p):
            new_coefs = [0 for _ in range(max(len(p.coefs), len(self.coefs)))]
            for i in range(len(self.coefs)): new_coefs[i] += self.coefs[i]
            for i in range(len(p.coefs)): new_coefs[i] += p.coefs[i]
            self.coefs = new_coefs
        else:
            self.coefs[0] += p
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
        return self

    def __sub__(self, p):
        return self + -1 * p

    def __isub__(self, p):
        self.coefs = (self - p).coefs
        return self

    def __str__(self):
        first_non_zero = 0
        for i in range(len(self.coefs)):
            if self.coefs[i] != 0:
                first_non_zero = i
                break
        output = f'{self.coefs[first_non_zero]}'
        if first_non_zero != 0:
            output += f'x^{first_non_zero}'
        for i in range(first_non_zero + 1, len(self.coefs)):
            abs_val = abs(self.coefs[i])
            if self.coefs[i] < 0:
                output += f' - {abs_val}x^{i}'
            elif self.coefs[i] > 0:
                output += f' + {abs_val}x^{i}'
        return output


class Methods:
    @staticmethod
    def approx_deriv(func, x, dx=0.001):
        x1 = x - dx/2
        x2 = x + dx/2
        y1 = func(x1)
        y2 = func(x2)
        dy = y2 - y1
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
        for zero in range(num_zeros):
            num_attempts = 0
            while True:
                x = random.uniform(min(x_range), max(x_range))
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
print(f'approximate zeros of polynomial {poly}: x={Methods.newton_approx(poly, num_zeros=3, x_range=(-1, 0))}')\

p1 = Polynomial([1, -2, 3, -13, 2, 4])
p2 = poly * p1


print(f'multiplying polynomial {poly} by {p1}: {p2}')
print(f'approximate zeros of polynomial {p2}: {Methods.newton_approx(p2, num_zeros=8, x_range=(-10, 10))}')


