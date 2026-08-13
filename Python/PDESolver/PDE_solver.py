import matplotlib.pyplot as plt
import numpy as np
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

class BoundaryCondition1D:
    def __init__(self, g, lower=True, upper=True, dirichlet = True):
        self.g = g
        self.lower = lower 
        self.upper = upper
        # Dirichlet specifies value
        self.dirichlet = dirichlet


def approx_derivs_central(U, t, dx, nx, bc: BoundaryCondition1D, approx_bounds=False, first=True, second=False, third=False, fourth=False):
    first_derivs = []
    second_derivs = []
    third_derivs = []
    fourth_derivs = []
    # Note this doesn't use actual x values, just their indices
    for xi in range(nx):
        # Approximate first derivative at x
        if first:
            if xi > 0 and xi < nx - 1:
                first_derivs.append(
                    (U[-1][xi+1] - U[-1][xi-1]) / (2 * dx)
                )
            else:
                if approx_bounds:
                    if xi == 0:  
                        first_derivs.append(
                            (U[-1][xi+2] - U[-1][xi]) / (2 * dx)
                        )
                    else:
                        first_derivs.append(
                            (U[-1][xi] - U[-1][xi-2]) / (2 * dx)
                        )
                else:
                    first_derivs.append(0)
        # Approximate second derivative at x
        if second:
            if xi > 0 and xi < nx - 1:
                second_derivs.append(
                    (U[-1][xi+1] - 2 * U[-1][xi] + U[-1][xi-1]) / dx**2
                )
            else:
                if approx_bounds:
                    if xi == 0:
                        second_derivs.append(
                            (U[-1][xi+2] - 2 * U[-1][xi+1] + U[-1][xi]) / dx**2
                        )
                    else:
                        second_derivs.append(
                            (U[-1][xi] - 2 * U[-1][xi-1] + U[-1][xi-2]) / dx**2
                        )
                else:
                    second_derivs.append(
                        0
                    )
        # Approximate third derivative at x
        if third:
            if xi > 1 and xi < nx - 2:
                third_derivs.append(
                    (U[-1][xi+2] - 2 * U[-1][xi+1] + 2 * U[-1][xi-1] - U[-1][xi-2]) / (2 * dx**3)
                )
            elif xi == nx - 2:
                third_derivs.append(
                    (U[-1][xi+1] - 2 * U[-1][xi] + 2 * U[-1][xi-2] - U[-1][xi-3]) / (2 * dx**3)
                )
            elif xi == 1:
                third_derivs.append(
                    (U[-1][xi+3] - 2 * U[-1][xi+2] + 2 * U[-1][xi] - U[-1][xi-1]) / (2 * dx**3)
                )
            else:
                if approx_bounds:
                    if xi == 0:
                        third_derivs.append(
                            (U[-1][xi+4] - 2 * U[-1][xi+3] + 2 * U[-1][xi+1] - U[-1][xi]) / (2 * dx**3)
                        )
                    else:
                        third_derivs.append(
                            (U[-1][xi] - 2 * U[-1][xi-1] + 2 * U[-1][xi-3] - U[-1][xi-4]) / (2 * dx**3)
                        )
                else:
                    third_derivs.append(0)
        # Approximate fourth derivative at x
        if fourth:
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
                if approx_bounds:
                    if xi == 0:
                        fourth_derivs.append(
                            (U[-1][xi+4] - 4 * U[-1][xi+3] + 6 * U[-1][xi+2] - 4 * U[-1][xi+1] + -U[-1][xi]) / dx**4
                        )
                    else:
                        fourth_derivs.append(
                            (U[-1][xi] - 4 * U[-1][xi-1] + 6 * U[-1][xi-2] - 4 * U[-1][xi-3] + -U[-1][xi-4]) / dx**4
                        )
                else:
                    fourth_derivs.append(0)
        
    res_tuple = ()
    if first:
        if approx_bounds == True and not bc.dirichlet:
            if bc.lower:
                first_derivs[0] = bc.g(0, t)
            if bc.upper:
                first_derivs[-1] = bc.g(dx * (nx-1), t)
        res_tuple += (first_derivs,)
    if second:
        res_tuple += (second_derivs,)
    if third:
        res_tuple += (third_derivs,)
    if fourth:
        res_tuple += (fourth_derivs,)

    return res_tuple
    
def solve_heat_eq(k, ic, bc: BoundaryCondition1D, dx, dt, nx = 100, nt = 100):
    x_grid = [dx * i for i in range(nx)]
    t_grid = [dt * i for i in range(nt)]

    # U should be a 2d array
    # 0th row contains u(x, 0) (initial conditions)
    # 1st row contains u(x, dt)
    # 2nd row contains u(x, 2*dt)
    U = [np.array([ic(x) for x in x_grid])]

    # For every timestep
    for ti in range(1, nt):
        new_U = []
        u_xx, u_xxxx = approx_derivs_central(U, ti * nt, dx, nx, bc, approx_bounds=False, first=False, second=True, third=False, fourth=True)
        for xi in range(nx):
            # This handles the boundaries
            if bc.lower and xi == 0:
                new_U.append(bc.g(xi * dx, ti * dt))
            elif bc.upper and xi == nx-1:
                new_U.append(bc.g(xi * dx, ti * dt))
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

"""
u_t + cu_x = 0

u(x, t + dt) ~ 
u(x, t) + u_t(x, t)dt + 1/2*u_tt(x, t)dt^2

u_t = -cu_x
u_tt = (u_t)_t = (-cu_x)_t = -c(u_t)_x = -c(-cu_x)_x = c^2u_xx
=>
u(x, t + dt) ~ 
u(x, t) -cu_x(x, t)dt + 1/2*c^2*u_xx(x, t)dt^2
"""
def solve_advec_eqn(c, ic, bc: BoundaryCondition1D, dx, dt, nx = 100, nt = 100):
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

        # We only approimate the derivatives on the bounds if we have von neumann (derivative) bc
        u_x, u_xx = approx_derivs_central(U, ti * nt, dx, nx, bc, approx_bounds=(not bc.dirichlet), first=True, second=True, third=False, fourth=False)
        
        for xi in range(nx):
            # If we are on the bounds and have dirichlet (value) bc
            if (xi == 0 or xi == nx-1) and bc.dirichlet:
                new_U.append(g(xi * dx, ti * dt))
            # If we are interior or boundary and have von neumann (derivative) bc, approx the value
            else:
                # u(x, t) -cu_x(x, t)dt + 1/2*c^2*u_xx(x, t)dt^2
                new_U.append(U[-1][xi] - c * u_x[xi] * dt + 0.5 * c**2 * u_xx[xi] * dt**2)
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


heat_bc = BoundaryCondition1D(lambda x, t: 0, lower=True, upper=True, dirichlet=True)
advec_bc = BoundaryCondition1D(lambda x, t: 1, lower = True, upper = False, dirichlet=False)

solve_heat_eq(1, lambda x: np.sin(4* np.pi * x), heat_bc, 0.01, 0.00001, nx=100, nt=2500)
# This needs work, weird spikes as t->inf
# solve_advec_eqn(1, lambda x: np.cos(4* np.pi * x)+1, advec_bc, 0.01, 0.00001, nx=100, nt=25000)

