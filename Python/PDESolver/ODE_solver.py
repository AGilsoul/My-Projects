import matplotlib.pyplot as plt
import numpy as np

class DiffEq:
    def dx_dt(t, x, args):
        return t
    
class exponential_ode(DiffEq): 
    def dx_dt(t, x, args):
        return x
    
class test_ode(DiffEq):
    def dx_dt(t, x, args):
        # x' = x1
        # x1' = x2
        # x2' = x3
        # x3' = -ax3 - bx2 -cx1 -dx
         
        # x'''' = -ax''' - bx'' - cx' - dx

        # dx/dt = x1
        # dx1/dt = x2
        # dx2/dt = x3
        # dx3/dt = -ax3 - bx2 - cx1 - dx

        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]

        return np.array([
            x[1],
            x[2],
            x[3],
            -a * x[3] - b * x[2] - c * x[1] - d * x[0]
        ])

class lotka_volterra(DiffEq):
    def dx_dt(t, x, args):
        alpha = args[0]
        beta = args[1]
        delta = args[2]
        gamma = args[3]
        return np.array([
            alpha * x[0] - beta * x[0] * x[1],
            delta * x[0] * x[1] - gamma * x[1]
        ])

def solve_ode_rk45(ode: DiffEq, x_0, t_0, h, h_tol, args, phase=False, iters=100):
    x = [x_0]
    t = [t_0]
    for i in range(iters):
        # Get the k values
        k1 = h * ode.dx_dt(t[-1], x[-1], args)
        k2 = h * ode.dx_dt(t[-1] + 1.0/4.0 * h, x[-1] + 1.0/4.0 * k1, args)
        k3 = h * ode.dx_dt(t[-1] + 3.0/8.0 * h, x[-1] + 3.0/32.0 * k1 + 9.0/32.0 * k2, args)
        k4 = h * ode.dx_dt(t[-1] + 12.0/13.0 * h, x[-1] + 1932.0/2197.0 * k1 - 7200.0/2197.0 * k2 + 7296.0/2197.0 * k3, args)
        k5 = h * ode.dx_dt(t[-1] + h, x[-1] + 439.0/216.0 * k1 - 8 * k2 + 3680.0/513.0 * k3 - 845.0/4104.0 * k4, args)
        k6 = h * ode.dx_dt(t[-1] + 1.0/2.0 * h, x[-1] - 8.0/27.0 * k1 + 2 * k2 - 3544.0/2565.0 * k3 + 1859.0/4104.0 * k4 - 11.0/40.0 * k5, args)

        # Get 4th and 5th order solutions
        x_k1 = x[-1] + 25.0/216.0 * k1 + 1408.0/2565.0 * k3 + 2197.0/4101.0 * k4 - 1.0/5.0 * k5
        z_k1 = x[-1] + 16.0/135.0 * k1 + 6656.0/12825.0 * k3 + 28561.0/56430.0 * k4 - 9.0/50.0 * k5 + 2.0/55.0 * k6

        # Get timestep from these solutions
        max_norm = max(abs(z_k1 - x_k1))
        s = np.pow(0.5, 0.25) * np.pow(h_tol / max_norm, 0.25)
        dt = s * h

        # Approximate step
        x.append(x[-1] + ode.dx_dt(t[-1], x[-1], args) * dt)
        t.append(t[-1] + dt)

    x = np.array(x).T
    t = np.array(t)

    if not phase:
        for i in range(len(x_0)):
            plt.plot(t, x[i], label='Approx')
    else:
        plt.plot(x[0], x[1], label = '2D Phase space')
    plt.legend()
    plt.show()

def equilibrium_points():
    return

# solve_ode_rk45(ode=lotka_volterra, 
#                x_0=np.array([10, 10]), 
#                t_0=0, 
#                h=0.1, 
#                h_tol=1.0e-8, 
#                args=[1.1, 0.4, 0.1, 0.4], 
#                phase=True,
#                iters=10000)

