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

# solve_ode_rk45(ode=lotka_volterra, 
#                x_0=np.array([10, 10]), 
#                t_0=0, 
#                h=0.1, 
#                h_tol=1.0e-8, 
#                args=[1.1, 0.4, 0.1, 0.4], 
#                phase=True,
#                iters=10000)


    
""" 
u_tt = c^2u_xx

u(x,0) = f(x)
u_t(x,0) = g(x)

u(0,t) = u(L,t) = h(t)

2nd degree Taylor approximation
u(x0 + dx, t0 + dt) ~ 
u(x0, t0) + u_x(x0, t0)dx + u_t(x0, t0)dt + 1/2*u_xx(x0, t0)dx^2 + 1/2*u_tt(x0, t0)dt^2 + u_xt(x0, t0)dxdt

This won't work for hyperbolic PDEs
---------------------------------------------------------
This will work for first order parabolic PDEs though
Heat eq'n

u_t = ku_xx
u(x, 0) = f(x)
u(0, t) = u(L, t) = 0

We only expand in time

u(x, t + dt) ~ 
u(x, t) + u_t(x, t)dt + 1/2*u_tt(x, t)dt^2

Note that u_t = ku_xx => u_tt = ku_xxt = ku_txx = k(u_t)xx = ku_xxxx

Now from the eq'n
u(x, t + dt) ~
u(x, t) + ku_xx(x, t)dt + k/2*u_xxxx(x, t)dt^2

Now for u_xx, we approximate
u_xx(x_i, t) ~ (u(x_i+1, t) - 2u(x_i, t) + u(x_i-1, t)) / dx^2

Now for u_xxxx, we approximate
u_xxxx(x_i, t) ~ (u(x_{i+2}, t) - 4u(x_{i+1}, t) + 6u(x_i, t) - 4u(x_{i-1}, t) + u(x_{i-2}, t)) / dx^4


At t = 0:
u_xx(x_i, 0) ~ (f(x_{i+1}) - 2f(x_i) + f(x_{i-1})) / dx^2
u_xxxx(x_i, 0) ~ (f(x_{i+2}) - 4f(x_{i+1}) + 6f(x_i) - 4f(x_{i-1}) + f(x_{i-2})) / dx^4

Finally:
u(x, t + dt) ~ 
u(x, t) + 
((f(x_{i+1}) - 2f(x_i) + f(x_{i-1})) / dx^2) * dt + 
1/2*((f(x_{i+2}) - 4f(x_{i+1}) + 6f(x_i) - 4f(x_{i-1}) + f(x_{i-2})) / dx^4) * dt^2

"""
def approx_derivs(U, dx, nx):
    second_derivs = []
    fourth_derivs = []
    # Note this doesn't use actual x values, just their indices
    for xi in range(nx):
        # Approximate second derivative at x
        if xi > 0 and xi < nx - 1:
            second_derivs.append(
                (U[-1][xi+1] - 2 * U[-1][xi] + U[-1][xi-1]) / dx**2
            )
        else:
            second_derivs.append(
                0
            )
        # Approximate fourth derivative at x
        if xi > 1 and xi < nx - 2:
            fourth_derivs.append(
                (U[-1][xi+2] - 4 * U[-1][xi+1] + 6 * U[-1][xi] - 4 * U[-1][xi-1] + U[-1][xi-2]) / dx**4
            )
        elif xi == nx - 2:
            (U[-1][xi+1] - 4 * U[-1][xi] + 6 * U[-1][xi-1] - 4 * U[-1][xi-2] + U[-1][xi-3]) / dx**4
        elif xi == 1:
            fourth_derivs.append(
                (U[-1][xi+3] - 4 * U[-1][xi+2] + 6 * U[-1][xi+1] - 4 * U[-1][xi] + -U[-1][xi-1]) / dx**4
            )
        else:
            fourth_derivs.append(0)
    return second_derivs, fourth_derivs
    
def solve_heat_eq(k, ic, bc, dx, dt, nx = 100, nt = 100):
    x_grid = [dx * i for i in range(nx)]
    t_grid = [dt * i for i in range(nt)]

    f = ic # initial condition
    g = bc # boundary condition

    # U should be a 2d array
    # 0th row contains u(x, 0) (initial conditions)
    # 1st row contains u(x, dt)
    # 2nd row contains u(x, 2*dt)
    U = [np.array([ic(x) for x in x_grid])]

    # For every timestep
    for ti in range(1, nt):
        new_U = []
        u_xx, u_xxxx = approx_derivs(U, dx, nx)
        for xi in range(nx):
            # This handles the boundaries
            if xi == 0 or xi == nx-1:
                new_U.append(g(xi * dx))
            else:
                # u(x, t) + u_xx(x, t)dt + 1/2*u_xxxx(x, t)dt^2
                new_U.append(U[-1][xi] + k * u_xx[xi] * dt + 0.5 * k * u_xxxx[xi] * dt**2)
        # print(np.array(new_U))
        U.append(np.array(new_U))

    X, T = np.meshgrid(x_grid, t_grid)

    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X=X, Y=T, Z=np.vstack(U), cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u (Temperature)')
    ax.set_facecolor('black')
    ax.set_zlim(-1, 1)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    plt.show()

solve_heat_eq(1, lambda x: np.sin(4* np.pi * x), lambda x: 0, 0.01, 0.00001, nx=100, nt=2500)


