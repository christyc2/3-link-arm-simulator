from src.arm import RoboArm
from src.visualize import simulate_cubic_path
import numpy as np

# Initialize a 3-link arm with custom lengths
arm = RoboArm(2, 1, 1)
# Start and end x, y coordinates and orientation (radians) for end effector
start = [arm.max_length, 0, 0] # Start with horizontal arm facing right
end = [1, 1, -np.pi/2]

try:
    # Calculate joint angles via IK given end effector's position and orientation
    theta_start = arm.inverse_kinematics(start[0], start[1], start[2])
    theta_end = arm.inverse_kinematics(end[0], end[1], end[2])

    # Simulate the 3-link arm
    simulate_cubic_path(theta_start, theta_end, steps=100)
    print("Simulation completed successfully.")

except ValueError as e:
    print(e)