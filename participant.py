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


if __name__ == "__main__":

    ranges_ctrnn = {
        # 'sensor_gains': [1, 5],
        'sensor_biases': [-3, 3],
        'sensor_taus': [0.1, 1],
        'sensor_weights': [-8, 8],
        # 'inter_gains': [1, 10],
        # 'inter_biases': [-3, 3],
        'inter_taus': [0.1, 1.0],
        'inter_weights': [-8, 8],
        # 'motor_gains': [1, 5],
        'motor_biases': [-3, 3],
        'motor_taus': [0.1, 1],
        'motor_weights': [-8, 8],
    }

    n_sensor_neurons = 2
    n_inter_neurons = 3
    n_motor_neurons = 6

    n_parameters = 2 * n_sensor_neurons + n_sensor_neurons * n_inter_neurons + \
                    1 * n_inter_neurons + n_inter_neurons * n_inter_neurons + \
                    2 * n_motor_neurons + n_inter_neurons * n_motor_neurons

    ind = np.random.uniform(-1, 1, n_parameters)

    participant = Participant(n_sensor_neurons, n_inter_neurons, n_motor_neurons)
    participant.set_nervous_system(ind, ranges_ctrnn)
    participant.nervous_system.display_information()

    trial_number = 0

    participant.reset_agent(trial_number)

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib

    matplotlib.use('TkAgg')

    fig = plt.figure()
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.plot([0], [0], [0], 'ro')
    ax.plot([-1, 1], [0, 0], [0, 0], '--', color='grey')
    ax.plot([0, 0], [-1, 1], [0, 0], '--', color='grey')
    ax.plot([0, 0], [0, 0], [-1, 1], '--', color='grey')
    ax.plot(participant.xh, participant.yh, participant.zh, 'o')

    print(participant.xh, participant.yh, participant.zh)

    hand = []

    step_size = 0.1
    time = np.arange(0, 2, step_size)

    for t in time:
        participant.nervous_system.sensor_neurons_step(step_size)
        participant.nervous_system.inter_neurons_step(step_size)
        participant.nervous_system.motor_neurons_step(step_size)
        participant.update_angles(step_size)
        f = participant.forward_kinematic_on_plane()
        if not f:
            print(t)
        hand.append([participant.xh, participant.yh, participant.zh])

    hand = np.array(hand)

    ax.plot(hand[:, 0], hand[:, 1], hand[:, 2], '.')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()