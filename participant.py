import numpy as np
from arm_three_links import KinematicArm
from network_architecture import NervousSystem


class Participant(KinematicArm):

    def __init__(self, n_sensor_neurons, n_inter_neurons, n_motor_neurons):
        super().__init__()
        self.nervous_system = NervousSystem(n_sensor_neurons, n_inter_neurons, n_motor_neurons)

    def set_nervous_system(self, individual, ranges):
        self.nervous_system.set_network_configuration(individual, ranges)

    def reset_agent(self, x_i, y_i, gamma_i):
        self.reset_arm(x_i, y_i, gamma_i)
        self.nervous_system.randomize_outputs(0.5, 0.5)

    def update_angles(self, step_size):
        self.theta1 += step_size * (self.nervous_system.motor_outputs[0] - self.nervous_system.motor_outputs[1])
        self.theta2 += step_size * (self.nervous_system.motor_outputs[2] - self.nervous_system.motor_outputs[3])
        self.theta3 += step_size * (self.nervous_system.motor_outputs[4] - self.nervous_system.motor_outputs[5])

