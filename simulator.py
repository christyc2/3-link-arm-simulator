from kinematics import *
from visualize import *

# Lengths of the links
l1, l2, l3 = 3, 2.0, 1.2  

# Calculate joint angles via IK given end effector's position and orientation
try:
    theta1, theta2, theta3 = inverse_kinematics(3, 2, -np.pi/2, l1, l2, l3) 
    # Calculate joint coordinates via FK given joint angles and link lengths
    x_coords, y_coords = forward_kinematics(theta1, theta2, theta3, l1, l2, l3)

    # Plot the 3-link arm
    plot_arm(x_coords, y_coords)

except ValueError as e:
    print("The position is not reachable with the given link lengths.")