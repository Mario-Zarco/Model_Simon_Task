import matplotlib.pyplot as plt
import numpy as np
from findiff import FinDiff
from scipy import integrate

from participant import Participant
from ctrnn import sigmoid
from network_settings import n_sensor_neurons, n_inter_neurons, n_motor_neurons
from network_settings import n_parameters


def fitness_function(individual):

    ranges = {
        # 'sensor_gains': [1, 5],
        'sensor_biases': [-8, -3],
        'sensor_weights': [-16, 16],
        # 'inter_gains': [1, 10],
        # 'inter_biases': [-3, 3],
        'inter_taus': [0.1, 1.0],
        'inter_weights': [-16, 16],
        # 'motor_gains': [1, 5],  # 0.1 1
        # 'motor_biases': [-3, 3],  # -3 3
        # 'motor_weights': [-16, 16],
        # 'output_gains': [1, 2]
        # 'synergy': [-5, 5]
    }

    participant = Participant(n_sensor_neurons, n_inter_neurons, n_motor_neurons)
    participant.set_nervous_system(individual, ranges)
    # print(participant.nervous_system.display_information())

    step_size = 0.1
    stop_time = 10.0
    time = np.arange(0, stop_time, step_size)
    N = stop_time / step_size

    targets = np.array([-0.20, 0.20])

    acc_total_fitness = 0

    x_i = -0.10
    y_i = 0.15
    gamma = np.linspace(0.8*np.pi, 1.2*np.pi, 3)

    target_y = 0.60 # 55

    decision = np.linspace(0.25, 0.55, 3)

    left_target = -0.30
    right_target = 0.10

    congruency_list = ["congruent", "incongruent", "neutral"]

    for congruency in congruency_list:

        for t_x in targets:

            target_x = x_i + t_x
            # print(target_x, target_y)

            for gamma_i in gamma:

                for decision_time in decision:

                    participant.reset_agent(x_i, y_i, gamma_i)

                    d_i = np.sqrt((participant.x_hand - target_x) ** 2 + (participant.y_hand - target_y) ** 2)

                    x_hand = []
                    y_hand = []

                    flag_correct = True

                    fitness_i = 0

                    for t in time:

                        if target_x < 0 and congruency == "neutral":

                            # LEFT
                            # d = np.sqrt((participant.x_hand - left_target) ** 2)
                            d = np.sqrt((participant.x_hand - left_target) ** 2 + (participant.y_hand - target_y) ** 2)
                            sensor_d = 10 / (1 + 10 * d)
                            participant.nervous_system.sensor_inputs[0] = sensor_d
                            participant.nervous_system.sensor_inputs[1] = 0

                        elif target_x > 0 and congruency == "neutral":

                            # RIGHT
                            # d = np.sqrt((participant.x_hand - right_target) ** 2)
                            d = np.sqrt((participant.x_hand - right_target) ** 2 + (participant.y_hand - target_y) ** 2)
                            sensor_d = 10 / (1 + 10 * d)
                            participant.nervous_system.sensor_inputs[1] = sensor_d
                            participant.nervous_system.sensor_inputs[0] = 0

                        else:

                            # LEFT
                            # d = np.sqrt((participant.x_hand - left_target) ** 2)
                            d = np.sqrt((participant.x_hand - left_target) ** 2 + (participant.y_hand - target_y) ** 2)
                            sensor_d = 10 / (1 + 10 * d)
                            participant.nervous_system.sensor_inputs[0] = sensor_d

                            # RIGHT
                            # d = np.sqrt((participant.x_hand - right_target) ** 2)
                            d = np.sqrt((participant.x_hand - right_target) ** 2 + (participant.y_hand - target_y) ** 2)
                            sensor_d = 10 / (1 + 10 * d)
                            participant.nervous_system.sensor_inputs[1] = sensor_d

                        participant.nervous_system.sensor_inputs[2] = participant.theta1
                        participant.nervous_system.sensor_inputs[3] = participant.theta2
                        participant.nervous_system.sensor_inputs[4] = participant.theta3

                        if congruency == "neutral":
                            participant.nervous_system.sensor_inputs[5] = 0
                        elif congruency == "congruent":
                            if target_x < 0:
                                participant.nervous_system.sensor_inputs[5] = -sigmoid(
                                    10 * (t - 2 * decision_time))
                            else:
                                participant.nervous_system.sensor_inputs[5] = sigmoid(
                                    10 * (t - 2 * decision_time))
                        else:
                            if target_x < 0:
                                participant.nervous_system.sensor_inputs[5] = -2.0 * sigmoid(
                                    20 * (t - 2 * decision_time )) + \
                                                                               1.0 * sigmoid(
                                    20 * (t - decision_time / 2))
                            else:
                                participant.nervous_system.sensor_inputs[5] = 2.0 * sigmoid(
                                    20 * (t - 2 * decision_time)) - \
                                                                              1.0 * sigmoid(
                                    20 * (t - decision_time / 2))

                        participant.nervous_system.sensor_neurons_step()
                        participant.nervous_system.inter_neurons_step(step_size)
                        participant.update_angles(step_size)
                        participant.arm_step()

                        if participant.theta1 < 0 or participant.theta1 > np.pi:
                            flag_correct = False
                            break
                        elif participant.theta2 < 0 or participant.theta2 > np.pi:
                            flag_correct = False
                            break
                        elif participant.theta3 < -np.pi/3 or participant.theta3 > np.pi/3:
                            flag_correct = False
                            break

                        x_hand = np.append(x_hand, participant.x_hand)
                        y_hand = np.append(y_hand, participant.y_hand)

                        d_f = np.sqrt((participant.x_hand - target_x) ** 2 + (participant.y_hand - target_y) ** 2)

                        fitness_i += ((t + step_size) / stop_time) * (1 - d_f / d_i)

                    if flag_correct:

                        fitness_d = fitness_i / N

                        d_dt = FinDiff(0, step_size, 1, acc=6)
                        dx_dt = d_dt(x_hand)
                        dy_dt = d_dt(y_hand)
                        v_arm = np.sqrt(dx_dt ** 2 + dy_dt ** 2 )
                        v_f = v_arm[-1]
                        v_max = np.max(v_arm)

                        fitness_v = 1 - v_f / v_max

                        d3_dt3 = FinDiff(0, step_size, 3, acc=6)
                        d3x_dt3 = d3_dt3(x_hand)
                        d3y_dt3 = d3_dt3(y_hand)
                        square_magnitude_jerk = d3x_dt3 ** 2 + d3y_dt3 ** 2
                        c_arm = integrate.simpson(square_magnitude_jerk, time)

                        fitness = 0.5 * fitness_d + 0.5 * fitness_v - 0.5 * c_arm

                    else:

                        fitness = 0

                    acc_total_fitness += fitness

    return acc_total_fitness / (len(congruency_list) * targets.size * gamma.size * decision.size)