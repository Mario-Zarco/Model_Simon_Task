'''
Madhavun Candadi's CTRNN library (Some methods have been modified)
https://github.com/madvn/CTRNN
'''

import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize, float64


@vectorize([float64(float64)])
def sigmoid(s):
    if -s > np.log(np.finfo(type(s)).max):
        return 0.0
    return 1 / (1 + np.exp(-s))


def inverse_sigmoid(o):
    return np.log(o / (1 - o))


class CTRNN:

    def __init__(self, size=2):
        self.size = size
        self.taus = np.zeros(size)
        self.biases = np.zeros(size)
        self.gains = np.zeros(size)
        self.weights = np.zeros((size, size))
        self.states = np.zeros(size)
        self.outputs = np.zeros(size)
        self.external_inputs = np.zeros(size)

        # rk4
        self.temp_states = np.zeros(size)
        self.temp_outputs = np.zeros(size)
        self.rk4_k1 = np.zeros(size)
        self.rk4_k2 = np.zeros(size)
        self.rk4_k3 = np.zeros(size)
        self.rk4_k4 = np.zeros(size)

    @property
    def taus(self): return self.__taus

    @property
    def biases(self): return self.__biases

    @property
    def gains(self): return self.__gains

    @property
    def states(self): return self.__states

    @property
    def outputs(self): return self.__outputs

    @property
    def weights(self): return self.__weights

    @property
    def external_inputs(self): return self.__external_inputs

    @taus.setter
    def taus(self, ts):
        if ts.size != self.size:
            raise Exception("Size mismatch error - vector size != network size")
        self.__taus = ts

    @biases.setter
    def biases(self, bs):
        if bs.size != self.size:
            raise Exception("Size mismatch error - vector size != network size")
        self.__biases = bs

    @gains.setter
    def gains(self, gs):
        if gs.size != self.size:
            raise Exception("Size mismatch error - vector size != network size ")
        self.__gains = gs

    @states.setter
    def states(self, ss):
        if ss.size != self.size:
            raise Exception("Size mismatch error - vector size != network size")
        self.__states = ss

    @outputs.setter
    def outputs(self, os):
        if os.size != self.size:
            raise Exception("Size mismatch error - vector size != network size")
        self.__outputs = os

    @weights.setter
    def weights(self, ws):
        if ws.size != self.size*self.size:
            raise Exception("Size mismatch error - matrix vector != network^2 size")
        self.__weights = ws

    @external_inputs.setter
    def external_inputs(self, exs):
        if exs.size != self.size:
            raise Exception("Size mismatch error - vector size != network size")
        self.__external_inputs = exs

    def randomize_states(self, lb, ub):
        self.states = np.random.uniform(lb, ub, size=self.size)
        self.outputs = sigmoid(self.gains*(self.states + self.biases))

    def randomize_outputs(self, lb, ub):
        self.outputs = np.random.uniform(lb, ub, size=self.size)
        self.states = inverse_sigmoid(self.outputs)/self.gains - self.biases

    def set_neuron_state(self, i, value):
        self.states[i] = value
        self.outputs[i] = sigmoid(self.gains[i] * (self.states[i] + self.biases[i]))

    def set_neuron_output(self, i, value):
        self.outputs[i] = value
        self.states[i] = inverse_sigmoid(self.outputs[i]) / self.gains[i] - self.biases[i]

    def euler_step_numpy(self, step_size):
        if self.external_inputs.size != self.size:
            raise Exception("Size mismatch error - vector size != network size")
        total_inputs = self.external_inputs + self.outputs.dot(self.weights)
        self.states += step_size * self.taus * (total_inputs - self.states)
        self.outputs = sigmoid(self.gains*(self.states + self.biases))
        # print(self.outputs)

    def euler_step(self, step_size):
        for i in range(self.size):
            total_input = self.external_inputs[i]
            for j in range(self.size):
                total_input += self.weights[j][i]*self.outputs[j]
            self.states[i] += step_size * self.taus[i] * (total_input - self.states[i])

        for i in range(self.size):
            self.outputs[i] = sigmoid(self.gains[i] * (self.states[i] + self.biases[i]))

    def set_center_crossing(self):
        for i in range(self.size):
            input_weights = 0
            for j in range(self.size):
                input_weights += self.weights[j][i]
            theta_star = - input_weights / 2
            self.biases[i] = theta_star

    def rk4_step(self, step_size):
        # First Step
        for i in range(self.size):
            total_input = self.external_inputs[i]
            for j in range(self.size):
                total_input += self.weights[j][i] * self.outputs[j]
            self.rk4_k1[i] = step_size * self.taus[i] * (total_input - self.states[i])
            self.temp_states[i] = self.states[i] + 0.5 * self.rk4_k1[i]
            self.temp_outputs[i] = sigmoid(self.gains[i] * (self.temp_states[i] + self.biases[i]))

        # Second Step
        for i in range(self.size):
            total_input = self.external_inputs[i]
            for j in range(self.size):
                total_input += self.weights[j][i] * self.temp_outputs[j]
            self.rk4_k2[i] = step_size * self.taus[i] * (total_input - self.temp_states[i])
            self.temp_states[i] = self.states[i] + 0.5 * self.rk4_k2[i]

        for i in range(self.size):
            self.temp_outputs[i] = sigmoid(self.gains[i] * (self.temp_states[i] + self.biases[i]))

        # Third Step
        for i in range(self.size):
            total_input = self.external_inputs[i]
            for j in range(self.size):
                total_input += self.weights[j][i] * self.temp_outputs[j]
            self.rk4_k3[i] = step_size * self.taus[i] * (total_input - self.temp_states[i])
            self.temp_states[i] = self.states[i] + self.rk4_k3[i]

        for i in range(self.size):
            self.temp_outputs[i] = sigmoid(self.gains[i] * (self.temp_states[i] + self.biases[i]))

        # Fourth Step
        for i in range(self.size):
            total_input = self.external_inputs[i]
            for j in range(self.size):
                total_input += self.weights[j][i] * self.temp_outputs[j]
            self.rk4_k4[i] = step_size * self.taus[i] * (total_input - self.temp_states[i])
            self.states[i] = self.states[i] + (1/6) * self.rk4_k1[i] + (1/3) * self.rk4_k2[i] + (1/3) * self.rk4_k3[i] + (1/6) * self.rk4_k4[i]
            self.outputs[i] = sigmoid(self.gains[i] * (self.states[i] + self.biases[i]))


if __name__ == "__main__":

    stop_time = 250
    step_size = 0.1
    time = np.arange(0, stop_time, step_size)

    network = CTRNN(2)
    network.taus = np.array([1, 1])
    network.gains = np.array([1, 1])
    network.biases = np.array([-2.75, -1.75])
    network.weights[0, 0] = 4.5
    network.weights[0, 1] = -1
    network.weights[1, 0] = 1
    network.weights[1, 1] = 4.5

    network.randomize_states(-0.5, 0.5)
    # network.randomize_outputs(0.1, 0.2)
    outputs = []

    for t in time:
        network.euler_step(step_size)
        # network.rk4_step(step_size)
        outputs.append([network.outputs[0],
                        network.outputs[1]])
    outputs = np.array(outputs)

    plt.figure(0)

    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(time, outputs[:, 0])
    plt.plot(time, outputs[:, 1])
    plt.title("Euler")
    network.randomize_states(-0.5, 0.5)
    # network.randomize_outputs(0.1, 0.2)
    outputs = []

    for t in time:
        # network.euler_step(step_size)
        network.rk4_step(step_size)
        outputs.append([network.outputs[0],
                        network.outputs[1]])
    outputs = np.array(outputs)

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(time, outputs[:, 0])
    plt.plot(time, outputs[:, 1])
    plt.title("Runge Kutta")

    plt.show()