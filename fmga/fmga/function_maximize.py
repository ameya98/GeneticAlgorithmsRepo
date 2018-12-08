'''
fmga
Genetic Algorithms - Objective Function Maximization
Author: Ameya Daigavane
Date: 15th April, 2018
'''

# External library dependencies
from random import randint, uniform
import numpy as np
import pathos.multiprocessing as mp


# Weighted choice function - each choice has a corresponding weight
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

    # Create random n-dimensional point within boundaries
    def __init__(self, associated_population=None, dimensions=2):

        if associated_population is None:
            self.associated_population = None
            self.boundaries = [(0, 100) for _ in range(dimensions)]
            self.mutation_range = 5
        else:
            self.associated_population = associated_population
            self.boundaries = associated_population.boundaries
            self.mutation_range = associated_population.mutation_range

        # Initialize coordinates uniformly random in range for each dimension
        self.coordinates = np.array([uniform(self.boundaries[dimension][0], self.boundaries[dimension][1]) for dimension in range(dimensions)])

        self.index = -1
        self.fitness = 0.0
        self.diversity = 0.0
        self.fitness_rank = -1
        self.diversity_rank = -1

    # Fitness score - objective function evaluated at the point
    def evaluate_fitness(self, eval_function=None):
        try:
            self.fitness = eval_function(*self.coordinates)
            return self.fitness
        except TypeError:
            print("function passed is invalid.")
            raise

    # Mutation operator
    def mutate(self):
        # Choose an index at random
        index = randint(0, np.size(self.coordinates) - 1)
        self.coordinates[index] += uniform(-self.mutation_range, self.mutation_range)

        # Ensure the point doesn't mutate out of range!
        self.coordinates[index] = min(self.boundaries[index][0], self.coordinates[index])
        self.coordinates[index] = max(self.boundaries[index][1], self.coordinates[index])

    def __repr__(self):
        return repr(self.coordinates)


# Parameter object for the Population parameters
class PopulationParameters:

    def __init__(self, dimensions, **kwargs):
        self.num_dimensions = dimensions
        self.population_size = kwargs.get('population_size', 60)
        self.boundaries = kwargs.get('boundaries')
        self.elite_fraction = kwargs.get('elite_fraction', 0.1)
        self.mutation_probability = kwargs.get('mutation_probability', 0.1)
        self.mutation_range = kwargs.get('mutation_range', 0.1)

        # Data-validation for parameters
        if self.elite_fraction > 1.0 or self.elite_fraction < 0.0:
            raise ValueError("Parameter 'elite_fraction' must be in range [0,1].")

        if self.mutation_probability > 1.0 or self.mutation_probability < 0.0:
            raise ValueError("Parameter 'mutation_probability' must be in range [0,1].")

        # Assign default boundaries if nothing passed
        if self.boundaries is None:
            self.boundaries = [(0, 100)] * self.num_dimensions
        else:
            try:
                # Default boundaries for missing parameters
                for dimension in range(len(self.boundaries), self.num_dimensions):
                    self.boundaries.append((0, 100))

                # Validate passed boundaries
                for dimension in range(len(self.boundaries)):
                    if float(self.boundaries[dimension][0]) > float(self.boundaries[dimension][1]):
                        raise ValueError("Incorrect value for boundary - min greater than max for range.")

            except TypeError:
                raise TypeError("Boundaries not passed correctly.")


# Population class and method definition
class Population:
    def __init__(self, objective_function=None, dimensions=None, **kwargs):

        if dimensions is None:
            try:
                # Use the objective function's number of arguments as dimensions
                dimensions = objective_function.__code__.co_argcount
            except TypeError:
                raise TypeError("Invalid function passed.")

        # Construct PopulationParameters object
        parameters = PopulationParameters(dimensions=dimensions, **kwargs)

        self.objective_function = objective_function
        self.num_dimensions = parameters.num_dimensions
        self.size = parameters.population_size
        self.elite_population_size = int(parameters.elite_fraction * self.size)
        self.mutation_probability = parameters.mutation_probability
        self.mutation_range = parameters.mutation_range
        self.boundaries = parameters.boundaries
        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
        self.mean_fitness = 0
        self.mean_diversity = 0
        self.mean_coordinates = np.zeros((self.num_dimensions, 1))
        self.num_iterations = 1

        # Multiprocessing defaults
        self.multiprocessing = kwargs.get('multiprocessing', False)
        self.processes = kwargs.get('processes')

        # Create points as Point objects
        self.points = []
        for pointnumber in range(self.size):
            point = Point(associated_population=self, dimensions=self.num_dimensions)
            self.points.append(point)
            self.points[pointnumber].index = pointnumber

        # If multiprocessing is enabled, create pool of processes.
        if self.multiprocessing:
            if self.processes is None:
                self.pool = mp.ProcessingPool()
            else:
                self.pool = mp.ProcessingPool(ncpus=self.processes)

            fitnesses = self.pool.map(lambda coordinates, func: func(*coordinates), [point.coordinates for point in self.points], [self.objective_function] * self.size)

            # Assign fitnesses to each point
            for index, point in enumerate(self.points):
                point.fitness = fitnesses[index]
        else:
            for point in self.points:
                point.evaluate_fitness(self.objective_function)

        # Evaluate fitness and diversity ranks
        self.__evaluate_fitness_ranks()
        self.__evaluate_diversity_ranks()

    # Evaluate the fitness rank of each point in the population
    def __evaluate_fitness_ranks(self):
        if not self.evaluated_fitness_ranks:
            self.mean_fitness = np.sum(point.fitness for point in self.points) / self.size

            # sort and assign ranks
            self.points.sort(key=lambda point: point.fitness, reverse=True)
            for rank_number in range(self.size):
                self.points[rank_number].fitness_rank = rank_number

            self.evaluated_fitness_ranks = True

    # Evaluate the diversity rank of each point in the population
    def __evaluate_diversity_ranks(self):
        if not self.evaluated_diversity_ranks:
            # Find mean coordinates
            self.mean_coordinates = np.sum(point.coordinates for point in self.points) / self.size

            for point in self.points:
                point.diversity = np.sum(np.abs(point.coordinates - self.mean_coordinates))

            self.mean_diversity = np.sum(point.diversity for point in self.points) / self.size

            self.points.sort(key=lambda point: point.diversity, reverse=True)
            for rank_number in range(self.size):
                self.points[rank_number].diversity_rank = rank_number

            self.evaluated_diversity_ranks = True

    # Generate the new population by breeding points
    def __breed(self):
        # Sort according to fitness rank
        self.points.sort(key=lambda point: point.fitness_rank)

        # Push all the really good points first (according to fitness)
        newpopulation = []
        for pointnumber in range(self.elite_population_size):
            newpopulation.append(self.points[pointnumber])

        # Assign weights to being selected for breeding
        weights = [1 / (1 + point.fitness_rank + point.diversity_rank) for point in self.points]

        # Randomly select for the rest and breed
        while len(newpopulation) < self.size:
            parent1 = weighted_choice(list(range(self.size)), weights)
            parent2 = weighted_choice(list(range(self.size)), weights)

            # Don't breed with yourself, dude!
            while parent1 == parent2:
                parent1 = weighted_choice(list(range(self.size)), weights)
                parent2 = weighted_choice(list(range(self.size)), weights)

            # Breed now
            child1, child2 = crossover(self.points[parent1], self.points[parent2])

            # Add the children
            newpopulation.append(child1)
            if len(newpopulation) < self.size:
                newpopulation.append(child2)

        # Re-assign to the new population
        self.points = newpopulation

        # Evaluate fitnesses of new population points
        if self.multiprocessing:
            # Reuse pool of processes
            fitnesses = self.pool.map(lambda coordinates, func: func(*coordinates), [point.coordinates for point in self.points], [self.objective_function] * self.size)

            # Assign fitnesses to each point
            for index, point in enumerate(self.points):
                point.fitness = fitnesses[index]
        else:
            for point in self.points:
                point.evaluate_fitness(self.objective_function)

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False

    # Mutate population randomly
    def __mutate(self):
        for point in self.points:
            mutate_probability = uniform(0, 1)
            if mutate_probability < self.mutation_probability:
                point.mutate()
                point.evaluate_fitness(self.objective_function)

                self.evaluated_fitness_ranks = False
                self.evaluated_diversity_ranks = False

    # Perform one iteration of breeding and mutation
    def iterate(self, verbose=0):
        # Breed
        self.__breed()

        # Mutate
        self.__mutate()

        # Find the new population's fitness and diversity ranks.
        self.__evaluate_fitness_ranks()
        self.__evaluate_diversity_ranks()

        # Print the population stats, if enabled
        if verbose == 1:
            print("Iteration", self.num_iterations, "complete.")
        elif verbose == 2:
            print("Iteration", self.num_iterations, "complete, with statistics:")
            print("Mean fitness =", self.mean_fitness)
            print("Mean L1 diversity =", self.mean_diversity)
            print()

        self.num_iterations += 1

    # Perform iterations sequentially
    def converge(self, iterations=15, verbose=0):
        for iteration in range(1, iterations + 1):
            self.iterate(verbose=verbose)

    # The point with best fitness is the estimate of point of maxima
    def best_estimate(self):
        best_point_fitness = float("-inf")
        best_point = None
        for point in self.points:
            if point.fitness > best_point_fitness:
                best_point_fitness = point.fitness
                best_point = point

        return best_point


# Crossover (breed) 2 points by swapping coordinates
def crossover(point1, point2):
    if point1.associated_population != point2.associated_population:
        raise ValueError("Points are from different populations.")

    child1 = Point(associated_population=point1.associated_population, dimensions=np.size(point1.coordinates))
    child2 = Point(associated_population=point2.associated_population, dimensions=np.size(point2.coordinates))

    splitpoint = randint(1, np.size(point1.coordinates))

    child1.coordinates = np.concatenate([point1.coordinates[:splitpoint], point2.coordinates[splitpoint:]])
    child2.coordinates = np.concatenate([point2.coordinates[:splitpoint], point1.coordinates[splitpoint:]])

    return child1, child2


# Wrapper to build a population and converge to function maxima, returning the best point as a Point object
def maximize(objective_function=None, dimensions=None, iterations=15, verbose=0, **kwargs):

    population = Population(objective_function=objective_function, dimensions=dimensions, **kwargs)
    population.converge(iterations=iterations, verbose=verbose)

    return population.best_estimate()


# Wrapper to build a population and converge to function minima, returning the best point as a Point object
def minimize(objective_function=None, dimensions=None, iterations=15, verbose=0, **kwargs):

    # Negative of the objective function
    def objective_function_neg(*args):
        return -objective_function(*args)

    # Minimize the function by maximizing the negative of the function.
    best_point = maximize(objective_function=objective_function_neg, dimensions=dimensions,
                          iterations=iterations, verbose=verbose, **kwargs)

    best_point.evaluate_fitness(objective_function)

    return best_point


# Helper to unpack arguments with shapes as given
def unpack(args, shapes):
    try:
        # Convert passed arguments to a numpy array
        np_args = np.array(args)
        index = 0
        unpacked_args = []

        # Step through the passed arguments and reshape them one-by-one
        for shape in shapes:
            currprod = 1
            try:
                for val in shape:
                    currprod *= val
            except TypeError:
                currprod *= shape
            finally:
                unpacked_args.append(np_args[index: index + currprod].reshape(shape))
                index += currprod

        if len(shapes) > 1:
            return unpacked_args
        else:
            return unpacked_args[0]

    except (TypeError, IndexError):
        raise

