# Genetic Algorithms - Objective Function Maximization #
# Author: Ameya Daigavane #
# Date: 15th April, 2018 #

# external library dependencies
from random import randint, uniform
import numpy as np


# a weighted choice function
def weighted_choice(choices, weights):
    normalized_weights = np.array([weight for weight in weights]) / np.sum(weights)
    threshold = uniform(0, 1)
    total = 1

    for index, normalized_weight in enumerate(normalized_weights):
        total -= normalized_weight
        if total < threshold:
            return choices[index]


# Point class and method definitions
class Point:

    # create random n-dimensional point within boundaries
    def __init__(self, associated_population=None, dimensions=2):

        if associated_population is None:
            self.associated_population = None
            self.boundaries = [(0, 100) for _ in range(dimensions)]
            self.mutation_range = 5
        else:
            self.associated_population = associated_population
            self.boundaries = associated_population.boundaries
            self.mutation_range = associated_population.mutation_range

        self.coordinates = np.array([uniform(self.boundaries[dimension][0], self.boundaries[dimension][1]) for dimension in range(dimensions)])

        self.index = -1
        self.fitness = 0.0
        self.diversity = 0.0
        self.fitness_rank = -1
        self.diversity_rank = -1

    # fitness score - objective function evaluated at the point
    def evaluate_fitness(self, eval_function=None):
        try:
            self.fitness = eval_function(*self.coordinates)
            return self.fitness
        except TypeError:
            print("function passed is invalid.")
            raise

    # mutation
    def mutate(self):
        # choose the index at random
        index = randint(0, np.size(self.coordinates) - 1)
        self.coordinates[index] += uniform(-self.mutation_range, self.mutation_range)

        # point shouldn't mutate out of range!
        self.coordinates[index] = min(self.boundaries[index][0], self.coordinates[index])
        self.coordinates[index] = max(self.boundaries[index][1], self.coordinates[index])

    def __repr__(self):
        return repr(self.coordinates)


# Population class and method definition
class Population:
    def __init__(self, objective_function=None, population_size=60, boundaries=None,
                 elite_fraction=0.1, mutation_probability=0.05, mutation_range=5, verbose=2, dimensions=None):

        if elite_fraction > 1.0 or elite_fraction < 0.0:
            raise ValueError("Parameter 'elite_fraction' must be in range [0,1].")

        if mutation_probability > 1.0 or mutation_probability < 0.0:
            raise ValueError("Parameter 'mutation_probability' must be in range [0,1].")

        if verbose not in [0, 1, 2]:
            raise ValueError("Parameter verbose must be one of 0, 1 or 2.")

        if dimensions is None:
            try:
                # use the function's number of arguments as dimensions
                self.num_dimensions = objective_function.__code__.co_argcount
            except TypeError:
                print("Invalid function passed.")
                raise
        else:
            self.num_dimensions = dimensions

        if boundaries is None:
            boundaries = []
            for dimension in range(self.num_dimensions):
                # default boundaries
                boundaries.append((0, 100))
        else:
            if type(boundaries) is not list:
                raise TypeError("boundaries not passed as a list.")
            else:
                for dimension in range(len(boundaries), self.num_dimensions):
                    # default boundaries
                    boundaries.append((0, 100))

                for dimension in range(len(boundaries)):
                    if type(boundaries[dimension]) is not tuple:
                        raise TypeError("boundary entry not passed as a tuple.")
                    else:
                        if float(boundaries[dimension][0]) > float(boundaries[dimension][1]):
                            raise ValueError("min greater than max in boundary entry.")

        self.points = []
        self.size = population_size
        self.objective_function = objective_function
        self.elite_population_size = int(elite_fraction * self.size)
        self.mutation_probability = mutation_probability
        self.mutation_range = mutation_range
        self.boundaries = boundaries
        self.verbose = verbose

        for pointnumber in range(self.size):
            point = Point(associated_population=self, dimensions=self.num_dimensions)
            point.evaluate_fitness(self.objective_function)
            self.points.append(point)
            self.points[pointnumber].index = pointnumber

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
        self.mean_fitness = 0
        self.mean_coordinates = np.zeros((self.num_dimensions, 1))
        self.mean_diversity = 0
        self.num_iterations = 1

        # evaluate the ranks
        self.__evaluate_fitness_ranks()
        self.__evaluate_diversity_ranks()

    # evaluate fitness rank of each point in population
    def __evaluate_fitness_ranks(self):
        if not self.evaluated_fitness_ranks:
            self.mean_fitness = np.sum(point.fitness for point in self.points) / self.size

            self.points.sort(key=lambda point: point.fitness, reverse=True)
            for rank_number in range(self.size):
                self.points[rank_number].fitness_rank = rank_number

            self.evaluated_fitness_ranks = True

    # evaluate diversity rank of each point in population
    def __evaluate_diversity_ranks(self):
        if not self.evaluated_diversity_ranks:
            # find mean coordinates
            self.mean_coordinates = np.sum(point.coordinates for point in self.points) / self.size

            for point in self.points:
                point.diversity = np.sum(np.abs(point.coordinates - self.mean_coordinates))

            self.mean_diversity = np.sum(point.diversity for point in self.points) / self.size

            self.points.sort(key=lambda point: point.diversity, reverse=True)
            for rank_number in range(self.size):
                self.points[rank_number].diversity_rank = rank_number

            self.evaluated_diversity_ranks = True

    # generate the new population by breeding points
    def __breed(self):
        # sort according to fitness rank
        self.points.sort(key=lambda point: point.fitness_rank)

        # push all the really good points first
        newpopulation = []
        for pointnumber in range(self.elite_population_size):
            newpopulation.append(self.points[pointnumber])

        # assign weights to being selected for breeding
        weights = [1 / (1 + point.fitness_rank + point.diversity_rank) for point in self.points]

        # randomly select for the rest and breed
        while len(newpopulation) < self.size:
            parent1 = weighted_choice(list(range(self.size)), weights)
            parent2 = weighted_choice(list(range(self.size)), weights)

            # don't breed with yourself, dude!
            while parent1 == parent2:
                parent1 = weighted_choice(list(range(self.size)), weights)
                parent2 = weighted_choice(list(range(self.size)), weights)

            # breed now
            child1, child2 = crossover(self.points[parent1], self.points[parent2])

            # evaluate fitnesses of children
            child1.evaluate_fitness(self.objective_function)
            child2.evaluate_fitness(self.objective_function)

            # add the children
            newpopulation.append(child1)

            if len(newpopulation) < self.size:
                newpopulation.append(child2)

        # assign the new population
        self.points = newpopulation
        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False

    # mutate population randomly
    def __mutate(self):
        for point in self.points:
            test_probability = uniform(0, 1)
            if test_probability < self.mutation_probability:
                point.mutate()
                point.evaluate_fitness(self.objective_function)

                self.evaluated_fitness_ranks = False
                self.evaluated_diversity_ranks = False

    # perform one iteration
    def iterate(self):
        # breed and mutate
        self.__breed()
        self.__mutate()

        # find the new ranks
        self.__evaluate_fitness_ranks()
        self.__evaluate_diversity_ranks()

        # print the stats
        if self.verbose == 1:
            print("Iteration", self.num_iterations, "complete.")
        elif self.verbose == 2:
            print("Iteration", self.num_iterations, "complete, with statistics:")
            print("Mean fitness =", self.mean_fitness)
            print("Mean L1 diversity =", self.mean_diversity)
            print()

        self.num_iterations += 1

    # perform the iterations sequentially
    def converge(self, iterations=15):
        for iteration in range(1, iterations + 1):
            self.iterate()

    # point with best fitness is the estimate of point of maxima
    def best_estimate(self):
        best_point_fitness = float("-inf")
        best_point = None
        for point in self.points:
            if point.fitness > best_point_fitness:
                best_point_fitness = point.fitness
                best_point = point

        return best_point


# crossover (breed) 2 points by swapping x-coordinates
def crossover(point1, point2):
    if point1.associated_population != point2.associated_population:
        raise ValueError("Points are from different populations.")

    child1 = Point(associated_population=point1.associated_population, dimensions=np.size(point1.coordinates))
    child2 = Point(associated_population=point2.associated_population, dimensions=np.size(point2.coordinates))

    splitpoint = randint(1, np.size(point1.coordinates))

    child1.coordinates = np.concatenate([point1.coordinates[:splitpoint], point2.coordinates[splitpoint:]])
    child2.coordinates = np.concatenate([point2.coordinates[:splitpoint], point1.coordinates[splitpoint:]])

    return child1, child2


def maximize(objective_function=None, population_size=60, boundaries=None, elite_fraction=0.1,
             mutation_probability=0.05, mutation_range=5, verbose=2, dimensions=None, iterations=15):

    population = Population(objective_function, population_size, boundaries, elite_fraction, mutation_probability, mutation_range, verbose, dimensions)
    population.converge(iterations)

    return population.best_estimate()

