import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from kinematics import *


def plot_arm(x, y):
    """
    Plot the 3-link planar arm based on the joint angles.
    
    Parameters:
    x (list): x-coordinates of the joints
    y (list): y-coordinates of the joints
    """
    if len(x) != 4 or len(y) != 4:
        raise ValueError("x and y must contain exactly 4 elements each, starting with the base.")
    
    # Plot the arm
    plt.figure()
    plt.scatter(x, y, c=['black','blue','red','green'], zorder=2)
    plt.plot(x, y, linestyle='-', c='black', linewidth=2, zorder=1)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.set_ylim(0, max(y) + 1)
    plt.set_aspect('equal')
    plt.title("3-Link Planar Arm")

    # Draw end effector
    dx = x[3] - x[2]
    dy = y[3] - y[2]
    length = np.hypot(dx, dy)
    perp_dx = -dy / length * 0.3 # 0.2 is half length of the perpendicular line
    perp_dy = dx / length * 0.3
    
    para_dx = -perp_dy / np.hypot(perp_dx, perp_dy) * 0.4
    para_dy = perp_dx / np.hypot(perp_dx, perp_dy) * 0.4

    plt.plot([x[3] - perp_dx - para_dx, x[3] - perp_dx, x[3] + perp_dx, x[3] + perp_dx - para_dx],
             [y[3] - perp_dy - para_dy, y[3] - perp_dy, y[3] + perp_dy, y[3] + perp_dy - para_dy],
             c='purple', linewidth=2, zorder=4)

    plt.grid(True)
    plt.show()

def simulate_cubic_path(theta_start, theta_end, l1, l2, l3, steps=100):
    """
    Animate a 3-link robot arm moving along a cubic joint path.
    theta_start, theta_end: in radians, shape = (3,)
    """

    # Cubic interpolation
    times = [0, 1]
    angles = np.array([theta_start, theta_end]).T
    splines = [CubicSpline(times, joint, bc_type='clamped') for joint in angles]

    t_vals = np.linspace(0, 1, steps)
    trajectory = np.array([[spline(t) for spline in splines] for t in t_vals])

    # Set up matplotlib figure
    fig, ax = plt.subplots()
    arm_line, = ax.plot([], [], 'o-', lw=3, c='black')
    eff_line, = ax.plot([], [], 'o-', lw=3, c='red')
    ax.set_xlim(-sum([l1,l2,l3]) - 1, sum([l1,l2,l3]) + 1)
    ax.set_ylim(-1, sum([l1,l2,l3]) + 1)
    ax.set_aspect('equal')
    ax.grid(True)

    def calculate_end_effector(x, y):
        dx = x[3] - x[2]
        dy = y[3] - y[2]
        length = np.hypot(dx, dy)
        perp_dx = -dy / length * 0.3 # 0.2 is half length of the perpendicular line
        perp_dy = dx / length * 0.3
        hypo = np.hypot(perp_dx, perp_dy)
        para_dx = -perp_dy / hypo * 0.4
        para_dy = perp_dx / hypo * 0.4

        eff_x_new = [x[3] - perp_dx - para_dx, x[3] - perp_dx, x[3] + perp_dx, x[3] + perp_dx - para_dx]
        eff_y_new = [y[3] - perp_dy - para_dy, y[3] - perp_dy, y[3] + perp_dy, y[3] + perp_dy - para_dy]
        return eff_x_new, eff_y_new

    def update(frame_num):
        thetas = trajectory[frame_num]
        x_new, y_new = forward_kinematics(thetas[0], thetas[1], thetas[2], l1, l2, l3)
        arm_line.set_data(x_new, y_new)
        eff_x, eff_y = calculate_end_effector(x_new, y_new)
        eff_line.set_data(eff_x, eff_y)
        return arm_line, eff_line

    ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    
    plt.title("3-Link Arm Simulator")
    plt.show()