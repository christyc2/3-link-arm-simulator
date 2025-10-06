import numpy as np

class RoboArm:
    def __init__(self, l1, l2, l3,):
        """
        Initializes a 3-link planar robotic arm.

        Parameters:
        l1 (float): length of link 1
        l2 (float): length of link 2
        l3 (float): length of link 3
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def max_length(self):
        return self.l1 + self.l2 + self.l3

    def inverse_kinematics(self, x_end, y_end, rho):
        """
        Calculate the inverse kinematics for a 3-link manipulator given the end effector position and orientation and link lengths.
        Returns a tuple of the joint angles in radians.
        
        Parameters:
        x_end (float): x-coordinate of the end effector
        y_end (float): y-coordinate of the end effector
        rho (float): end effector orientation from +x-axis
        """
        x_prime = x_end - self.l3 * np.cos(rho)
        y_prime = y_end - self.l3 * np.sin(rho)
        A = (x_prime**2 + y_prime**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        
        # Check if the position is reachable
        if abs(A) > 1:
            raise ValueError("The position is not reachable with the given link lengths.")
        
        # Calculate theta1, theta2, and theta3
        theta2 = -np.arccos(A) # Two possible solutions
        theta1 = np.arctan2(y_prime, x_prime) - np.arcsin( self.l2 * np.sin(theta2) / np.sqrt(x_prime**2 + y_prime**2) )
        # Want the solution with theta1 in the range [0, pi]
        if theta1 < 0:
            theta2 *= -1
            theta1 = np.arctan2(y_prime, x_prime) - np.arcsin( self.l2 * np.sin(theta2) / np.sqrt(x_prime**2 + y_prime**2) )
        theta3 = rho - theta1 - theta2
        
        return theta1, theta2, theta3

    def forward_kinematics(self, theta1, theta2, theta3):
        """
        Returns the x-coordinates and y-cooordinates of each joint as tuples.
        Calculate the forward kinematics for a 3-link manipulator given the joint angles and link lengths.   
        Require t1, t2, and t3 are the joint angles relative to the previous link, in radians, in [-pi, pi].

        Parameters:
        t1 (float): angle of joint 1 
        t2 (float): angle of joint 2
        t3 (float): angle of joint 3
        l1 (float): length of link 1
        l2 (float): length of link 2
        l3 (float): length of link 3
        """
        x0, y0 = 0, 0
        x1 = x0 + self.l1 * np.cos(theta1)
        y1 = y0 + self.l1 * np.sin(theta1)
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        x3 = x2 + self.l3 * np.cos(theta1 + theta2 + theta3)
        y3 = y2 + self.l3 * np.sin(theta1 + theta2 + theta3)
        
        return [x0, x1, x2, x3], [y0, y1, y2, y3]