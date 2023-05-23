from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

sigma = 10.0
rho = 28.0
beta = 8.0/3.0
y0 = [50, 50, 50]
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


def draw_ellipsoid(fig, ax):
    m = min(1.0, 1.0/sigma, beta/(2*sigma))
    coefs = (m / (3 * beta * rho), m * sigma / (3 * beta * rho**2), m* sigma / (3 * beta * rho**2))
    rx, ry, rz = 1/np.sqrt(coefs)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    z += np.ones_like(z) * 2 * rho
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='g', alpha=0.1)


def heat_map(fig, ax):
    min_val, max_val = -800, 801
    step = 20
    x1 = np.arange(min_val, max_val, step)
    y1 = np.arange(1, 2)
    z1 = np.arange(min_val, max_val, step)

    x2 = np.arange(min_val, max_val, step)
    y2 = np.arange(min_val, max_val, step)
    z2 = np.arange(1, 2)

    x1, y1, z1 = np.meshgrid(x1, y1, z1, indexing='xy')
    x2, y2, z2 = np.meshgrid(x2, y2, z2, indexing='xy')

    points = []
    points = fill_heat_pts(fill_heat_pts(points, x1, y1, z1), x2, y2, z2)

    points = np.array(points).T
    points_plotted = ax.scatter(points[0], points[1], points[2],
                        c=points[3], cmap='plasma', alpha=0.1)


def fill_heat_pts(cur_pts, x, y, z):
    for xi in range(len(x)):
        print(xi)
        for yi in range(len(x[xi])):
            for zi in range(len(x[xi][yi])):
                res = v(x[xi][yi][zi], y[xi][yi][zi], z[xi][yi][zi])
                cur_pts.append([x[xi][yi][zi], y[xi][yi][zi], z[xi][yi][zi], res])
    return cur_pts


def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # heat_map(fig, ax)
    draw_ellipsoid(fig, ax)
    solve_ode(fig, ax)
    plt.show()
    return


if __name__ == '__main__':
    main()
