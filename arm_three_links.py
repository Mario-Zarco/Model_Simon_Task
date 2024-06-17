import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def arc_tan_2(y, x):
    return np.arctan(y / x) + np.sign(y) * (1 - np.sign(x)) * np.pi / 2


class KinematicArm:

    def __init__(self):
        self.l1 = 0.31
        self.l2 = 0.32
        self.l3 = 0.05

        # Initial Conditions
        self.x_hand, self.y_hand = 0, 0
        self.x_wrist, self.y_wrist = 0, 0
        self.x_elbow, self.y_elbow = 0, 0
        self.theta1, self.theta2, self.theta3 = 0, 0, 0

    def reset_arm(self, xi, yi, gamma):
        self.x_hand = xi
        self.y_hand = yi
        self.inverse_kinematics(gamma)
        # print(ik)

    def forward_kinematics(self):
        self.x_elbow = self.l1 * np.cos(self.theta1)
        self.x_wrist = self.x_elbow + self.l2 * np.cos(self.theta1 + self.theta2)
        self.x_hand = self.x_wrist + self.l3 * np.cos(self.theta1 + self.theta2 + self.theta3)
        self.y_elbow = self.l1 * np.sin(self.theta1)
        self.y_wrist = self.y_elbow + self.l2 * np.sin(self.theta1 + self.theta2)
        self.y_hand = self.y_wrist + self.l3 * np.sin(self.theta1 + self.theta2 + self.theta3)

    def arm_step(self):
        self.forward_kinematics()

    def inverse_kinematics(self, gamma):
        # gamma = self.theta1 + self.theta2 + self.theta3
        x_w = self.x_hand - self.l3 * np.cos(gamma)
        y_w = self.y_hand - self.l3 * np.sin(gamma)
        c2 = (x_w**2 + y_w**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if -1 <= c2 <= 1:
            s2 = np.sqrt(1 - c2**2)
            self.theta2 = arc_tan_2(s2, c2)
            s1 = ((self.l1 + self.l2 * c2) * y_w - self.l2 * s2 * x_w) / (x_w**2 + y_w**2)
            c1 = ((self.l1 + self.l2 * c2) * x_w + self.l2 * s2 * y_w) / (x_w**2 + y_w**2)
            self.theta1 = arc_tan_2(s1, c1)
            self.theta3 = gamma - self.theta1 - self.theta2
            return True
        else:
            return False


if __name__ == "__main__":

    arm = KinematicArm()
    arm.reset_arm(-0.1, 0.1, 1.1*np.pi)

    fig = plt.figure()
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(True)
    ax.axvline(0, ls='--', color='k')
    ax.axhline(0, ls='--', color='k')

    x_e = arm.l1 * np.cos(arm.theta1)
    y_e = arm.l1 * np.sin(arm.theta1)

    x_w = x_e + arm.l2 * np.cos(arm.theta1 + arm.theta2)
    y_w = y_e + arm.l2 * np.sin(arm.theta1 + arm.theta2)

    x_h = x_w + arm.l3 * np.cos(arm.theta1 + arm.theta2 + arm.theta3)
    y_h = y_w + arm.l3 * np.sin(arm.theta1 + arm.theta2 + arm.theta3)

    print(arm.x_hand, arm.y_hand)
    arm.forward_kinematics()
    print(arm.x_hand, arm.y_hand)

    ax.plot([0, x_e, x_w, x_h], [0, y_e, y_w, y_h], '-')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.1, 0.9)

    plt.show()
