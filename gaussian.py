import numpy as np


class Gaussian:

    def __init__(self):
        self.gaussian_flag = 0
        self.gx1 = 0
        self.gx2 = 0

    def gaussian_random(self, mean, variance):
        if not self.gaussian_flag:
            self.generate_normals()
            self.gaussian_flag = 1
            return np.sqrt(variance) * self.gx1 + mean
        else:
            self.gaussian_flag = 0
            return np.sqrt(variance) * self.gx2 + mean

    def generate_normals(self):
        while True:
            v1 = np.random.uniform(-1, 1)
            v2 = np.random.uniform(-1, 1)
            s = v1 * v1 + v2 * v2
            if s < 1.0 and s != 0:
                break
        d = np.sqrt((-2 * np.log(s)) / s)
        self.gx1 = v1 * d
        self.gx2 = v2 * d

    def random_unit_array(self, array):
        r = 0.0
        for i in range(0, array.size, 1):
            array[i] = self.gaussian_random(0, 1)
            r += array[i] * array[i]
        r = np.sqrt(r)
        for i in range(0, array.size, 1):
            div = array[i] / r
            array[i] = div