import random


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

    @staticmethod
    def __newton_approx_recurse__(func, cur_x, num_iterations, iter_count):
        if iter_count < num_iterations:
            deriv = Methods.approx_deriv(func, cur_x)
            x1 = Methods.linear_zero(deriv, (cur_x, func(cur_x)))
            return Methods.__newton_approx_recurse__(func, x1, num_iterations, iter_count + 1)
        else:
            return cur_x

    @staticmethod
    # coefficients in order of powers from least to greatest
    def calc_poly(coefs, x):
        return sum([coefs[i]*x**i for i in range(len(coefs))])

    @staticmethod
    def mult_poly(coefs_1, coefs_2):
        new_coefs = [0 for _ in range((len(coefs_1))+(len(coefs_2)) - 1)]
        for i in range(len(coefs_1)):
            for j in range(len(coefs_2)):
                cur_power = i + j
                new_coefs[cur_power] += coefs_1[i] * coefs_2[j]
        return new_coefs


def function(x):
    return Methods.calc_poly([0, 2, 5, 3], x)


val = -7
print(f'polynomial 3x^3 + 5x^2 + 2x - 7 at x={val}: {Methods.calc_poly([-7, 2, 5, 3], val)}')
print(f'approximate zeros of polynomial 3x^3 + 5x^2 + 2x - 7: x={Methods.newton_approx(function, num_zeros=3, x_range=(-1, 0))}')\

test_poly = Methods.mult_poly([-7, 2, 5, 3], [1, -2, 3, -13, 2, 4])


def new_func(x):
    return Methods.calc_poly(test_poly, x)


print(f'multiplying polynomial 3x^3 + 5x^2 + 2x - 7 by 4x^5 + 2x^4 + 13x^3 + 3x^2 - 2x + 1: {test_poly}')
print(f'approximate zeros of polynomial 12x^8 + 26x^7 + 57x^6 + 50x^5 + 21x^4 - 92x^3 - 20x^2 + 16x - 7: {Methods.newton_approx(new_func, num_zeros=8, x_range=(-10, 10))}')


