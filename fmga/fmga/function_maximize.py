# Genetic Algorithms - Objective Function Maximization #
# Author: Ameya Daigavane #
# Date: 15th April, 2018 #

# libraries for the genetic algorithm
from random import randint, uniform
from math import sin, exp, pi

# objective function to maximise
def objective_function(x, y):
    return (10 * sin((5 * pi * x)/(2 * 100)) ** 2) * (10 * sin((5 * pi * y)/(2 * 100)) ** 2) * exp(-(x + y)/100)
    # return 1/(x + y + 1)
    # return x + y if x > 50 else y - x


# a weighted choice function
def weighted_choice(choices, weights):
    normalized_weights = [weight/sum(weights) for weight in weights]
    # print(normalized_weights)
    threshold = uniform(0, 1)
    total = 1
    for index, normalized_weight in enumerate(normalized_weights):
        total -= normalized_weight
        if total < threshold:
            return choices[index]


# Point2D class and method definitions
class Point2D:
    # create random 2D point within boundaries [x_min, x_max] and [y_min, y_max]
    def __init__(self, x_min=0, x_max=100, y_min=0, y_max=100, mutation_range=5):

        if x_min > x_max:
            raise ValueError("x_min greater than x_max.")

        if y_min > y_max:
            raise ValueError("y_min greater than y_max.")

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.x = uniform(x_min, x_max)
        self.y = uniform(y_min, y_max)

        self.index = -1
        self.fitness = 0.0
        self.diversity = 0.0
        self.fitness_rank = -1
        self.diversity_rank = -1
        self.mutation_range = mutation_range

    # fitness score - objective function evaluated at the point
    def evaluate_fitness(self, eval_function=None):
        try:
            self.fitness = eval_function(self.x, self.y)
            return self.fitness
        except TypeError:
            print("function passed is invalid.")
            raise

    # mutation
    def mutate(self):
        index = randint(0, 1)
        if index == 0:
            self.x += uniform(-self.mutation_range, self.mutation_range)

            # point shouldn't mutate out of range!
            self.x = min(self.x, self.x_max)
            self.x = max(self.x, self.x_min)
        else:
            self.y += uniform(-self.mutation_range, self.mutation_range)

            # point shouldn't mutate out of range!
            self.y = min(self.y, self.y_max)
            self.y = max(self.y, self.y_min)


# Population class and method definition
class population2D:
    def __init__(self, population_size=60, objective_function=None, elite_fraction=0.1, mutation_probability=0.05,
                 x_min=0, x_max=100, y_min=0, y_max=100, mutation_range=5, verbose=2):

        if x_min > x_max:
            raise ValueError("Parameter x_min greater than x_max.")

        if y_min > y_max:
            raise ValueError("Parameter y_min greater than y_max.")

        if elite_fraction > 1.0 or elite_fraction < 0.0:
            raise ValueError("Parameter 'elite_fraction' must be in range [0,1].")

        if mutation_probability > 1.0 or mutation_probability < 0.0:
            raise ValueError("Parameter 'mutation_probability' must be in range [0,1].")

        if verbose not in [0, 1, 2]:
            raise ValueError("Parameter verbose must be one of 0, 1 or 2.")

        self.points = []
        self.size = population_size
        self.objective_function = objective_function
        self.elite_population_size = int(elite_fraction * self.size)
        self.mutation_probability = mutation_probability
        self.coordinate_range = ((x_min, x_max), (y_min, y_max))
        self.verbose = verbose

        for pointnumber in range(self.size):
            point = Point2D(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, mutation_range=mutation_range)
            self.points.append(point)
            self.points[pointnumber].index = pointnumber

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
        self.mean_fitness = 0
        self.mean_diversity = 0
        self.x_mean = 0
        self.y_mean = 0
        self.num_iterations = 1

        # evaluate the ranks
        self.__evaluate_fitness_ranks()
        self.__evaluate_diversity_ranks()

    # evaluate fitness rank of each point in population
    def __evaluate_fitness_ranks(self):
        if not self.evaluated_fitness_ranks:

            self.mean_fitness = 0
            for point in self.points:
                point.evaluate_fitness(self.objective_function)
                self.mean_fitness += point.fitness

            self.mean_fitness /= self.size
            self.points.sort(key=lambda point: point.fitness, reverse=True)

            for rank_number in range(self.size):
                self.points[rank_number].fitness_rank = rank_number

            self.evaluated_fitness_ranks = True

    # evaluate diversity rank of each point in population
    def __evaluate_diversity_ranks(self):
        if not self.evaluated_diversity_ranks:
            # find mean x and y coordinates
            self.x_mean = 0
            self.y_mean = 0

            for point in self.points:
                self.x_mean += point.x
                self.y_mean += point.y

            self.x_mean /= self.size
            self.y_mean /= self.size

            self.mean_diversity = 0
            for point in self.points:
                point.diversity = (abs(point.x - self.x_mean) + abs(point.y - self.y_mean))
                self.mean_diversity += point.diversity

            self.mean_diversity /= self.size
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

    # perform the iterations
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
    child1 = Point2D(x_min=point1.x_min, x_max=point2.x_max, y_min=point1.y_min, y_max=point1.y_max,
                     mutation_range=point1.mutation_range)
    child2 = Point2D(x_min=point2.x_min, x_max=point2.x_max, y_min=point2.y_min, y_max=point2.y_max,
                     mutation_range=point2.mutation_range)

    child1.x = point1.x
    child1.y = point2.y

    child2.x = point2.x
    child2.y = point1.y

    return child1, child2


if __name__ == "__main__":
    # initialize the population
    population = population2D(population_size=60, objective_function=objective_function)

    # print the initial stats
    print("Initial Population")
    print("Mean fitness =", population.mean_fitness)
    print("Mean L1 diversity =", population.mean_diversity)
    print()

    # breed and mutate for 15 iterations
    population.converge(iterations=15)

    # get the best_estimate
    best_point = population.best_estimate()

    # print the stats
    print("Function Maximum Estimate =", best_point.fitness)
    print("Function Maximum Position Estimate =", "(" + str(best_point.x) + ", " + str(best_point.y) + ")")
