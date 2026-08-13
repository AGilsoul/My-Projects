import random
import matplotlib.pyplot as plt
from typing import Union


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
    def __init__(self, coefs: list, var='x'):
        super().__init__(var)
        if isinstance(coefs, int):
            coefs = [coefs]
        coefs = [float(x) for x in coefs]
        self.coefs = coefs

    def __call__(self, x):
        return sum([self.coefs[i] * x ** i for i in range(len(self.coefs))])

    def shorten(self):
        for i in reversed(range(len(self.coefs))):
            if self.coefs[i] == 0 and i != 0 and i == len(self.coefs) - 1:
                self.coefs.pop(i)
                break
        if isinstance(self.coefs, int):
            self.coefs = [self.coefs]

    def __add__(self, p):
        if type(self) == type(p):
            new_coefs = [0 for _ in range(max(len(p.coefs), len(self.coefs)))]
            for i in range(len(self.coefs)): new_coefs[i] += self.coefs[i]
            for i in range(len(p.coefs)): new_coefs[i] += p.coefs[i]
        else:
            new_coefs = self.coefs
            new_coefs[0] += p
        result = Polynomial(new_coefs, self.var)
        result.shorten()
        return result

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
        result = Polynomial(new_coefs, self.var)
        result.shorten()
        return result

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
        result = self + p * -1
        result.shorten()
        return result

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

    def is_zero(self):
        return len(self.coefs) == 1 and self.coefs[0] == 0 

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
    def newton_approx(func, num_zeros=1, x_range=(-1, 1), rand=True, start_point=None, num_iterations=100, max_attempts=100, verbose=False):
        zeros = []
        range_min = min(x_range)
        range_max = max(x_range)
        for zero in range(num_zeros):
            num_attempts = 0
            while True:
                if rand and start_point is None:
                    x = random.uniform(range_min, range_max)
                else:
                    x = start_point
                for i in range(num_iterations):
                    deriv = Methods.approx_deriv(func, x)
                    x = Methods.linear_zero(deriv, (x, func(x)))
                num_attempts += 1
                if x not in zeros:
                    if verbose:
                        print('found zero')
                    zeros.append(x)
                    break
                if num_attempts >= max_attempts:
                    if verbose:
                        print('exceeded attempts')
                    break
        return sorted(zeros)

    @staticmethod
    def newton_poly_iterative(p: Polynomial, num_zeros=1, known_zeros=None, x_range=(-1, 1), num_iterations=100, max_attempts=100, verbose=False):
        # Set up my roots and a copy of the original polynomial that I can manipulate
        roots = []
        remaining_p = Polynomial(p.coefs, var=p.var)

        # Divide out any known roots
        if known_zeros is not None:
            for zero in known_zeros:
                poly_divisor = Polynomial([-zero, 1], var=p.var)
                print(f'Dividing out {poly_divisor} (known root at {zero})')
                print()
                remaining_p = Methods.poly_div(remaining_p, poly_divisor)
                roots.append(zero)

        # Determine the step size between starting points
        step_size = (x_range[1] - x_range[0]) / (num_zeros * 100)
        start_x = [x_range[0] + step_size * i for i in range(num_zeros * 100)]

        initial_iterations = int(num_iterations * 1e3)
        # Find each root one at a time, divide it out
        
        known_zeros = []
        
        for i in range(num_zeros - len(known_zeros)):
            search_x = (x_range[1] + x_range[0]) / 2.0
            low_val_x = x_range[0]
            low_val = remaining_p(low_val_x)
            upper_val_x = x_range[1]
            upper_val = remaining_p(upper_val_x)


            print(f'Finding starting x...')
            # Now use binary method to find a good starting point, if can't find one, just start in middle
            lower_index = 0
            upper_index = -1
            upper_changed = False
            # First make sure we start with bounds of opposite sign
            while remaining_p(start_x[lower_index]) * remaining_p(start_x[upper_index]) > 0 and lower_index < len(start_x) and upper_index > -len(start_x):
                if upper_changed:
                    lower_index += 1
                    upper_changed = False
                else:
                    upper_index -= 1
                    upper_changed = True
                if upper_index == lower_index:
                    break                
                # print(f'set bounds to p({start_x[lower_index]})={remaining_p(start_x[lower_index])}, p({start_x[upper_index]})={remaining_p(start_x[upper_index])}')

                low_val_x = start_x[lower_index]
                upper_val_x = start_x[upper_index]
                low_val = remaining_p(low_val_x)
                upper_val = remaining_p(upper_val_x)

            if low_val * upper_val <= 0:
                print(f'Binary searching for a good start point for a root...')
                for i in range(num_iterations):
                    mid_val_x = (upper_val_x + low_val_x) / 2.0
                    mid_val = remaining_p(mid_val_x)
                    if mid_val * upper_val > 0:
                        upper_val_x = mid_val_x
                        upper_val = mid_val
                    else:
                        low_val_x = mid_val_x
                        low_val = mid_val
                    # print(f'lower x: {low_val_x}, upper x: {upper_val_x}')
                search_x = mid_val_x
                    
            print(f'Starting x is {search_x}')

            print(f'Approximating with {remaining_p}')
            cur_root = Methods.newton_approx(remaining_p, 1, rand=False, start_point=search_x, num_iterations=num_iterations, max_attempts=max_attempts, verbose=verbose)[0]

            poly_divisor = Polynomial([-cur_root, 1], var=p.var)
            remaining_p = Methods.poly_div(remaining_p, poly_divisor)
            print(f'Dividing out {poly_divisor} (found root at {cur_root})')
            print()
            roots.append(cur_root)
        return sorted(roots)

    @staticmethod
    def poly_div(p1: Polynomial, p2: Polynomial, rem=False) -> Union[Polynomial, tuple[Polynomial, Polynomial]]:
        """Computes p1/p2

        Args:
            p1 (Polynomial): Dividend polynomial.
            p2 (Polynomial): Divisor polynomial.
            rem (bool, optional): Set to True to return remainder as well. Defaults to False.

        Returns:
            Union[Polynomial, tuple[Polynomial, Polynomial]]: Quotient polynomial, or tuple containing quotient polynomial and remainder polynomial if rem set to True.
        """
        quotient = Polynomial([0], var=p1.var)
        # I have my polynomials in the form c_0 + c_1x + c_2x^2 + c_3x^3... and so on
        # So I have to go in reversed range
        remaining_dividend = p1
        while True:
            # First, figure by what we have to multiply the divisor to get up to the highest term of the dividend
            power_diff = len(remaining_dividend.coefs) - len(p2.coefs)
            if power_diff < 0 or remaining_dividend.is_zero():
                break
            mult_diff = remaining_dividend.coefs[-1] / p2.coefs[-1] 
            coefs = [0 for _ in range(power_diff + 1)]
            coefs[-1] = mult_diff
            factor_poly = Polynomial(coefs, var=p1.var)
            quotient += factor_poly

            poly_to_subtract = p2 * factor_poly
            remaining_dividend -= poly_to_subtract

        return quotient


def main():
    p1 = Polynomial([-4, 0, 1])
    p2 = Polynomial([2, 1])
    print(p1)
    print(p2)
    print(Methods.poly_div(p1, p2))

if __name__ == '__main__':
    main()
