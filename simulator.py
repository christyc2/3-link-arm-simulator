from kinematics import inverse_kinematics
from visualize import simulate_cubic_path
import numpy as np

# Lengths of the links
l1, l2, l3 = 2, 1, 1
# Start and end x, y coordinates and orientation (radians) for end effector
start = [sum([l1,l2,l3]), 0, 0] # Start with horizontal arm facing right
end = [1, 1, -np.pi/2]

try:
    # Calculate joint angles via IK given end effector's position and orientation
    theta_start = inverse_kinematics(start[0], start[1], start[2], l1, l2, l3)
    theta_end = inverse_kinematics(end[0], end[1], end[2], l1, l2, l3)

    # Simulate the 3-link arm
    simulate_cubic_path(theta_start, theta_end, l1, l2, l3, steps=100)
    print("Simulation completed successfully.")

except ValueError as e:
    print(e)