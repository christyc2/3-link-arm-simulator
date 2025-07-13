import numpy as np
import matplotlib.pyplot as plt

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
    plt.ylim(0, max(y) + 1)
    plt.gca().set_aspect('equal', adjustable='box')
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