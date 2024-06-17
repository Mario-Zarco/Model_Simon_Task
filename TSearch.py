'''
Adapted from Randall Beer's C++ TSearch library
https://rdbeer.pages.iu.edu/#_ga=2.158273470.500028861.1631741976-437507196.1630555352

Multiprocessing inspired by Madhavun Candadi's Python Evolutionary Search library
https://github.com/madvn/stochsearch

'''

import time
import numpy as np
from pathos.multiprocessing import ProcessPool
from gaussian import Gaussian


MIN_SEARCH_VALUE = -1.0
MAX_SEARCH_VALUE = 1.0


def map_search_parameters(x, min, max, clipmin=-1.0e99, clipmax=1.0e99):
    m = (max - min) / (MAX_SEARCH_VALUE - MIN_SEARCH_VALUE)
    b = min - m * MIN_SEARCH_VALUE
    return clip(m * x + b, clipmin, clipmax)


def linear_scale_factor(min, max, avg, FMultiple):
    if min > (FMultiple * avg - max) / (FMultiple - 1):
        delta = max - avg
        if delta > 0:
            return (FMultiple - 1) * avg / delta
        else:
            return 0.0
    else:
        delta = avg - min
        if delta > 0:
            return avg / delta
        else:
            return 0.0


def probabilistic_choice(prob):
    if np.random.uniform(0, 1) <= prob:
        return 1
    else:
        return 0


def clip(x, min_, max_):
    if x > min_:
        temp = x
    else:
        temp = min_
    if temp < max_:
        return temp
    else:
        return max_


__tsearch_process_pool = None


class TSearch:

    def __init__(self, evol_parameters):
        # check for required keys
        required_keys = ['population_size', 'genotype_size', 'fitness_function',
                         'max_exp_offspring', 'select_mode', 'elitist_fraction',
                         'crossover_probability', 'crossover_mode', 'mutation_variance',
                         'n_processes']
        for key in required_keys:
            if key not in evol_parameters.keys():
                raise Exception('Argument evol_parameters does not contain the following required key: ' + key)

        # required_keys = ['']
        self.population_size = evol_parameters['population_size']
        self.genotype_size = evol_parameters['genotype_size']
        self.fitness_function = evol_parameters['fitness_function']
        self.max_exp_offspring = evol_parameters['max_exp_offspring']
        self.select_mode = evol_parameters['select_mode']  # string
        self.elitist_fraction = evol_parameters['elitist_fraction']
        self.crossover_probability = evol_parameters['crossover_probability']
        self.crossover_mode = evol_parameters['crossover_mode']  # string
        self.mutation_variance = evol_parameters['mutation_variance']
        assert self.fitness_function, "Invalid fitness function"

        # self.min_search_value = -1.0
        # self.max_search_value = 1.0

        self.max_performance = 0
        self.min_performance = 0
        self.avg_performance = 0
        self.var_performance = 0
        self.best_performance = -1
        self.best_individual = np.zeros(self.genotype_size)
        self.gaussian = Gaussian()
        self.n_gen = 0

        # self.n_processes = evol_parameters.get('n_processes', None)
        self.population = np.random.uniform(MIN_SEARCH_VALUE, MAX_SEARCH_VALUE,
                                            (self.population_size, self.genotype_size))
        self.fitness = np.zeros(self.population_size)
        self.performance = np.zeros(self.population_size)
        # self.evaluate_population()

        self.n_processes = evol_parameters.get('n_processes', None)

        # creating the global process bool to be used across all generations
        global __tsearch_process_pool
        __tsearch_process_pool = ProcessPool(self.n_processes)
        time.sleep(0.5)

    def do_search(self, max_gen):
        self.evaluate_population()
        self.update_population_statistics()
        while self.n_gen < max_gen:
            self.n_gen += 1
            self.reproduce_population_genetic_algorithm()
            self.update_population_statistics()
            self.display_population_statistics()

    def evaluate_population(self): # start=0):
        global __tsearch_process_pool
        # self.performance = np.asarray(__tsearch_process_pool.map(self.evaluate_individual, np.arange(self.population_size)))
        if __tsearch_process_pool:
            self.performance = np.asarray(__tsearch_process_pool.map(self.evaluate_individual, np.arange(self.population_size)))
        else:
            __tsearch_process_pool = ProcessPool(self.n_processes)
            self.performance = np.asarray(__tsearch_process_pool.map(self.evaluate_individual, np.arange(self.population_size)))

    def evaluate_individual(self, individual_index):
        perf = self.fitness_function(self.population[individual_index, :])
        if perf < 0:
            return 0
        else:
            return perf
        # return 0 if perf < 0 else perf

    def update_population_statistics(self):
        # Update Min and Max Performance as necessary
        best_index = np.argmax(self.performance)
        self.max_performance = self.performance[best_index]
        self.min_performance = np.amin(self.performance)

        # Update total
        total = np.sum(self.performance)
        #  Update Average Performance (with protection from possible numerical error)
        self.avg_performance = total / self.population_size
        if self.avg_performance < self.min_performance:
            self.avg_performance = self.min_performance
        if self.avg_performance > self.max_performance:
            self.avg_performance = self.max_performance
        d = self.performance - self.avg_performance
        # Variance
        if self.population_size > 1:
            self.var_performance = np.sum(d * d) / (self.population_size - 1)
        else:
            self.var_performance = 0
        if self.max_performance > self.best_performance:
            self.best_performance = self.max_performance
            self.best_individual = self.population[best_index, :]

    def reproduce_population_genetic_algorithm(self):
        self.update_population_fitness()
        # Number of elite individuals in the new population
        elite_population = int(np.floor(self.elitist_fraction * self.population_size + 0.5))
        # Select the rest of the population using Baker's stochastic universal sampling
        temp_population = np.copy(self.population)
        sum = 0
        rand = np.random.uniform(0, 1)
        i = 0
        j = elite_population
        while i < self.population_size and j < self.population_size:
            sum += (self.population_size - elite_population) * self.fitness[i]
            while rand < sum:
                self.population[j] = np.copy(temp_population[i])
                rand += 1
                j += 1
            i += 1
        # Randomly shuffle the non-elite parents in preparation crossover
        if self.crossover_probability > 0:
            for i in range(elite_population, self.population_size, 1):
                k = np.random.randint(i, self.population_size)
                temp_individual = np.copy(self.population[k])
                self.population[k] = np.copy(self.population[i])
                self.population[i] = np.copy(temp_individual)
        # Apply mutation or crossover to each non-elite parent and compute the child's performance
        i = elite_population
        while i < self.population_size:
            # Perform crossover with probability crossprob
            if probabilistic_choice(
                    self.crossover_probability) and i < self.population_size - 1:
                parent1 = np.copy(self.population[i])
                # parent2 = self.population[i + 1]
                if self.crossover_mode == 'UNIFORM':
                    self.uniform_crossover(self.population[i], self.population[i + 1])
                elif self.crossover_mode == 'TWO_POINT':
                    self.two_point_crossover(self.population[i], self.population[i + 1])
                # IF the child is the same as the first parent after crossover, mutate it
                if np.array_equal(self.population[i], parent1):
                    self.mutate_individual(self.population[i])
                i += 1
            else:
                # Otherwise, perform mutation
                self.mutate_individual(self.population[i])
                i += 1
        # Evaluate the new population
        self.evaluate_population()  # (elite_population)

    def update_population_fitness(self):
        self.sort_population()
        # FITNESS_PROPORTIONATE
        if self.select_mode == "FITNESS_PROPORTIONATE":
            m = linear_scale_factor(self.min_performance, self.max_performance, self.avg_performance,
                                    self.max_exp_offspring)
            self.fitness = m * (self.performance - self.avg_performance) + self.avg_performance
            total = np.sum(self.fitness)
            self.fitness = self.fitness / total
        elif self.select_mode == "RANK_BASED":
            for i in range(0, self.population_size, 1):
                # self.fitness[i] = (self.max_exp_offspring + (2.0 - 2.0*self.max_exp_offspring)*((i - 1.0)/(self.population_size - 1))) / self.population_size
                self.fitness[i] = (self.max_exp_offspring + (2.0 - 2.0 * self.max_exp_offspring) * (i / self.population_size)) / self.population_size

    def sort_population(self):
        # self.population[np.]
        self.population = self.population[np.argsort(self.performance)[::-1]]
        self.performance = self.performance[np.argsort(self.performance)[::-1]]

    def uniform_crossover(self, parent1, parent2):
        cross_points = np.arange(0, self.genotype_size, 1)
        for i in range(0, cross_points.size - 1, 1):
            if probabilistic_choice(0.5):
                for j in range(cross_points[i], cross_points[i + 1], 1):
                    temp = np.copy(parent1[j])
                    parent1[j] = np.copy(parent2[j])
                    parent2[j] = np.copy(temp)

        if probabilistic_choice(0.5):
            for j in range(cross_points[cross_points.size - 1], self.genotype_size, 1):
                temp = np.copy(parent1[j])
                parent1[j] = np.copy(parent2[j])
                parent2[j] = np.copy(temp)

    def two_point_crossover(self, parent1, parent2):
        cross_points = np.arange(0, self.genotype_size, 1)
        i1 = np.random.randint(0, cross_points.size)
        i2 = i1
        while i2 == i1:
            i2 = np.random.randint(0, cross_points.size)
        if i1 > i2:
            t = i1
            i1 = i2
            i2 = t
        for i in range(cross_points[i1], cross_points[i2], 1):
            temp = np.copy(parent1[i])
            parent1[i] = np.copy(parent2[i])
            parent2[i] = np.copy(temp)

    def mutate_individual(self, individual):
        magnitude = self.gaussian.gaussian_random(0.0, self.mutation_variance)
        temp = np.zeros(self.genotype_size)
        self.gaussian.random_unit_array(temp)
        # Constraint vector is not used here
        for i in range(0, self.genotype_size, 1):
            individual[i] = clip(individual[i] + magnitude * temp[i], MIN_SEARCH_VALUE, MAX_SEARCH_VALUE)

    def display_population_statistics(self):
        print("Generation:", self.n_gen, ": Best =", self.best_performance,
              ", Average: ", self.avg_performance, ", Variance: ", self.var_performance)

    '''
    Extension (Hillclimbing method)
    '''

    def reproduce_population_hillclimbing(self):
        self.update_population_fitness()
        # Select the rest of the population using Baker's stochastic universal sampling
        parent_population = np.zeros((self.population_size, self.genotype_size))
        parent_performance = np.zeros(self.population_size)
        i = 0
        j = 0
        sum = 0
        rand = np.random.uniform(0, 1)
        while i < self.population_size and j < self.population_size:
            sum += self.population_size * self.fitness[i]
            while rand < sum:
                parent_population[j] = np.copy(self.population[i])
                parent_performance[j] = np.copy(self.performance[i])
                rand += 1
                j += 1
            i += 1
        # Replace the current population with the parent population
        self.population = np.copy(parent_population)
        # Produce the new population by mutating each parent
        for i in range(0, self.population_size, 1):
            self.mutate_individual(self.population[i])
        # Evaluate the children
        self.evaluate_population()
        # Restore each parent whose child's performance in worse
        for i in range(0, self.population_size, 1):
            if parent_performance[i] > self.performance[i]:
                self.population[i] = np.copy(parent_population[i])
                self.performance[i] = np.copy(parent_performance[i])

