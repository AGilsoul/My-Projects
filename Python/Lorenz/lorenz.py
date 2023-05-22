from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

sigma = 10.0
rho = 28.0
beta = 8.0/3.0
y0 = [10, 10, 10]
eval_span = [0, 40]


def diff(t, ic):
    x, y, z = ic[0], ic[1], ic[2]
    dx = -sigma * x + sigma * y
    dy = rho * x - y - x * z
    dz = -beta * z + x * y
    return dx, dy, dz


def v(x, y, z):
    return x**2 + y**2 + (z - 2 * rho)**2


def v_dot(x, y, z):
    return -2 * (rho * x**2 + y**2 + (z - rho)**2 - rho**2)


def solve_ode(fig, ax):
    ivp = solve_ivp(diff, eval_span, y0, method='LSODA', dense_output=True)
    res = ivp.y
    x = res[0]
    y = res[1]
    z = res[2]
    t = ivp.t
    ax.plot(x, y, z)


def heat_map(fig, ax):
    min_val, max_val = -200, 200
    step = 5
    x1 = np.arange(min_val, max_val, step)
    y1 = np.arange(1, 2)
    z1 = np.arange(min_val, max_val, step)

    x2 = np.arange(min_val, max_val, step)
    y2 = np.arange(min_val, max_val, step)
    z2 = np.arange(1, 2)
    points = []
    for xi in x1:
        print(xi)
        for yi in y1:
            for zi in z1:
                res = v(xi, yi, zi)
                points.append([xi, yi, zi, res])
    for xi in x2:
        print(xi)
        for yi in y2:
            for zi in z2:
                res = v(xi, yi, zi)
                points.append([xi, yi, zi, res])

    points = np.array(points).T
    points = ax.scatter(points[0], points[1], points[2],
                        c=points[3], cmap='plasma', alpha=0.01)


def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # heat_map(fig, ax)
    solve_ode(fig, ax)
    plt.show()
    return


if __name__ == '__main__':
    main()
