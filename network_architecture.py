import numpy as np
from ctrnn import CTRNN, sigmoid, inverse_sigmoid
from map_utils_numba import map_search_parameters


class NervousSystem:

    def __init__(self, n_sensor_neurons, n_inter_neurons, n_motor_neurons):
        self.n_sensor_neurons = n_sensor_neurons
        self.n_inter_neurons = n_inter_neurons
        self.n_motor_neurons = n_motor_neurons
        self.size = n_sensor_neurons + n_inter_neurons + n_motor_neurons

        self.sensor_inputs = np.zeros(self.n_sensor_neurons)
        self.sensor_states = np.zeros(self.n_sensor_neurons)
        self.sensor_taus = np.zeros(self.n_sensor_neurons)
        self.sensor_gains = np.zeros(self.n_sensor_neurons)
        self.sensor_biases = np.zeros(self.n_sensor_neurons)
        self.sensor_weights = np.zeros((self.n_sensor_neurons, self.n_inter_neurons))
        self.n_sensor_weights = self.sensor_weights.size
        self.sensor_outputs = np.zeros(self.n_sensor_neurons)

        self.inter_neurons = CTRNN(self.n_inter_neurons)
        self.n_inter_weights = self.inter_neurons.weights.size

        self.motor_states = np.zeros(self.n_motor_neurons)
        self.motor_taus = np.zeros(self.n_motor_neurons)
        self.motor_gains = np.zeros(self.n_motor_neurons)
        self.motor_biases = np.zeros(self.n_motor_neurons)
        self.motor_weights = np.zeros((self.n_inter_neurons, self.n_motor_neurons))
        self.n_motor_weights = self.motor_weights.size
        self.motor_outputs = np.zeros(self.n_motor_neurons)

        self.n_weights = self.n_sensor_weights + self.n_inter_weights + self.n_motor_weights

        self.limits = {
            'sensor_gains': n_sensor_neurons,
            'sensor_biases': n_sensor_neurons - 1,
            'sensor_taus': n_sensor_neurons,
            'sensor_weights': self.n_sensor_weights,
            'inter_gains': n_inter_neurons,
            'inter_biases': n_inter_neurons,
            'inter_taus': n_inter_neurons,
            'inter_weights': self.n_inter_weights,
            'motor_gains': n_motor_neurons,
            'motor_biases': n_motor_neurons,
            'motor_taus': n_motor_neurons,
            'motor_weights': self.n_motor_weights
        }

    def set_network_configuration(self, individual, ranges):
        p_limit = 0
        for param, range_ in ranges.items():
            # print(param, range_, self.limits[param])
            c_limit = p_limit + self.limits[param]
            # print(p_limit, c_limit)
            if param == 'sensor_gains':
                # print('sensor_gains')
                self.sensor_gains = map_search_parameters(individual[p_limit: c_limit], range_[0], range_[1])
            elif param == 'sensor_biases':
                # print('sensor_biases')
                self.sensor_biases = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1])
            elif param == 'sensor_taus':
                self.sensor_taus = map_search_parameters(individual[p_limit: c_limit], range_[0], range_[1])
            elif param == 'sensor_weights':
                # print('sensor_weights')
                self.sensor_weights = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1]).reshape((self.n_sensor_neurons, self.n_inter_neurons))
            elif param == 'inter_gains':
                # print('inter_gains')
                self.inter_neurons.gains = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1])
            elif param == 'inter_biases':
                # print('inter_biases')
                self.inter_neurons.biases = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1])
            elif param == 'inter_taus':
                # print('inter_taus')
                self.inter_neurons.taus = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1])
            elif param == 'inter_weights':
                # print('inter_weights')
                self.inter_neurons.weights = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1]).reshape((self.n_inter_neurons, self.n_inter_neurons))
            elif param == 'motor_gains':
                # print('motor_gains')
                self.motor_gains = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1])
            elif param == 'motor_biases':
                # print('motor_biases')
                self.motor_biases = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1])
            elif param == 'motor_taus':
                # print('motor_biases')
                self.motor_taus = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1])
            elif param == 'motor_weights':
                # print('motor_weights')
                self.motor_weights = map_search_parameters(individual[p_limit:c_limit], range_[0], range_[1]).reshape((self.n_inter_neurons, self.n_motor_neurons))
            p_limit = c_limit
        self.inter_neurons.set_center_crossing()
        self.inter_neurons.gains = np.ones(self.n_inter_neurons)

    def sensor_neurons_step(self):
        self.sensor_outputs[0:-1] = sigmoid(self.sensor_inputs[0:-1] + self.sensor_biases) # [0:-1])
        self.sensor_outputs[-1] = self.sensor_inputs[-1]

    def inter_neurons_step(self, step_size):
        self.inter_neurons.external_inputs = np.dot(self.sensor_outputs, self.sensor_weights)
        # self.inter_neurons.euler_step(step_size)
        self.inter_neurons.rk4_step(step_size)
        self.motor_outputs = self.inter_neurons.outputs

    def motor_neurons_step(self):
        self.motor_outputs = sigmoid(np.dot(self.inter_neurons.outputs, self.motor_weights))

    def randomize_outputs(self, lb, ub):
        self.inter_neurons.randomize_outputs(lb, ub)

    def display_information(self):
        print("----------------------------------------------------------------")
        print("Nervous System ")
        print("----------------------------------------------------------------")
        print("Sensor gains: ")
        print(self.sensor_gains)
        print("Sensor biases: ")
        print(self.sensor_biases)
        print("Sensor taus: ")
        print(self.sensor_taus)
        print("Sensor weights: ")
        print(self.sensor_weights)
        print("Inter gains: ")
        print(self.inter_neurons.gains)
        print("Inter biases: ")
        print(self.inter_neurons.biases)
        print("Inter taus: ")
        print(self.inter_neurons.taus)
        print("Inter weights: ")
        print(self.inter_neurons.weights)
        print("Motor gains: ")
        print(self.motor_gains)
        print("Motor biases: ")
        print(self.motor_biases)
        print("Motor taus: ")
        print(self.motor_taus)
        print("Motor weights: ")
        print(self.motor_weights)
        print("----------------------------------------------------------------")


if __name__ == "__main__":

    ranges = {
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
        'motor_weights': [-16, 16],
    }

    n_sensor_neurons = 1
    n_inter_neurons = 3
    n_motor_neurons = 2

    n_parameters = 2 * n_sensor_neurons + n_sensor_neurons * n_inter_neurons + \
                   1 * n_inter_neurons + n_inter_neurons * n_inter_neurons + \
                   2 * n_motor_neurons + n_inter_neurons * n_motor_neurons

    print("Number of parameters", n_parameters)

    ind = np.random.uniform(-1, 1, n_parameters)

    neural_network = NervousSystem(n_sensor_neurons, n_inter_neurons, n_motor_neurons)
    neural_network.set_network_configuration(ind, ranges)
    neural_network.display_information()
