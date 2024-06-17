n_sensor_neurons = 6
n_inter_neurons = 6
n_motor_neurons = 0

n_parameters = 1 * (n_sensor_neurons - 1) + n_sensor_neurons * n_inter_neurons + \
                1 * n_inter_neurons + n_inter_neurons * n_inter_neurons + \
                0 * n_motor_neurons + n_inter_neurons * n_motor_neurons