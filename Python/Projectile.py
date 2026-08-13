import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib import animation


anim_target_X = []
anim_interceptor_X = []
anim_target_vel = [0.0, 0.0, 0.0]
anim_interceptor_vel = [0.0, 0.0, 0.0]
past_target_X = []
past_interceptor_X = []
int_acc = 10.0

target_vector_new_dir = [0, 1]
target_dir_change_time = 0
target_dir_dt = 0

last_int_diff = 0.0
int_integral = 0.0

p = 0.1
i = 0.0
d = 1.0

max_output = 5.0

def interceptor_acc(target_X, interceptor_X, dt):
    global last_int_diff, int_integral, p, i, d, max_output
    diff_proportion = target_X - interceptor_X
    int_integral += diff_proportion
    diff_derivative = (diff_proportion - last_int_diff) / dt
    last_int_diff = diff_proportion
    acc_vec = p * diff_proportion + i * int_integral + d * diff_derivative
    acc_mag = np.linalg.norm(acc_vec)
    if acc_mag > max_output:
        acc_vec = acc_vec / acc_mag * max_output
    return acc_vec

def interceptor_vel(target_X, interceptor_X, dt):
    global anim_interceptor_vel
    anim_interceptor_vel += interceptor_acc(target_X, interceptor_X, dt) * dt
    return anim_interceptor_vel

def interceptor_pos(target_X, interceptor_X, dt):
    global anim_interceptor_X, past_interceptor_X
    # anim_interceptor_X += interceptor_vel(target_X, interceptor_X, dt) * dt
    past_interceptor_X.append(np.copy(anim_interceptor_X))
    anim_interceptor_X += 2 * (target_X - interceptor_X) / np.linalg.norm(target_X - interceptor_X) * dt

def target_acc(target_X, interceptor_X, dt):
    return

def target_vel(target_X, interceptor_X, dt):
    return np.array([0, 0, 1])

def target_pos(target_X, interceptor_X, dt):
    global anim_target_X, past_target_X
    past_target_X.append(np.copy(anim_target_X))
    anim_target_X += target_vel(target_X, interceptor_X, dt) * dt



def figure_update(t, ax, dt, dims, labels):
    global anim_target_X, anim_interceptor_X, anim_target_vel, anim_interceptor_vel, past_target_X, past_interceptor_X
    ax.cla()
    # target
    target_pos(anim_target_X, anim_interceptor_X, dt)
    print(anim_target_X)
    past_target_arr = np.array(past_target_X).T
    print(past_target_arr)
    ax.plot(past_target_arr[0][-2000:], past_target_arr[1][-2000:], past_target_arr[2][-2000:], c='blue')
    ax.scatter(anim_target_X[0], anim_target_X[1], anim_target_X[2], s=10, c='blue')
    # Interceptor
    interceptor_pos(anim_target_X, anim_interceptor_X, dt)
    past_interceptor_arr = np.array(past_interceptor_X).T
    ax.plot(past_interceptor_arr[0][-2000:], past_interceptor_arr[1][-2000:], past_interceptor_arr[2][-2000:], c='red')
    ax.scatter(anim_interceptor_X[0], anim_interceptor_X[1], anim_interceptor_X[2], s=10, c='red')
    xlims = dims[0]
    ylims = dims[1]
    zlims = dims[2]
    
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ax.set_zlim(zlims[0], zlims[1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    

def intercept(target_X: np.array, X_0: np.array, max_t=100, dt=0.01, dims=[(0, 10), (0, 10), (0, 10)], labels=['x', 'y', 'z']):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    global anim_target_X, anim_interceptor_X, past_target_X, past_interceptor_X
    
    anim_target_X = np.array(target_X, dtype=float)
    anim_interceptor_X = np.array(X_0, dtype=float)
    past_target_X = [anim_target_X]
    past_interceptor_X = [anim_interceptor_X]
    anim = animation.FuncAnimation(fig, figure_update, fargs=(ax, dt, dims, labels),
                                   interval=1, blit=False)
    # plt.legend()
    plt.show()
    plt.close()
    return

intercept([0, 0, 0], [0, 10, 0])