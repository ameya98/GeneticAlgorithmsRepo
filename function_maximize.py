# Genetic Algorithms - Objective Function Maximization #

# libraries for the genetic algorithm
from random import randint, uniform
from math import sin, exp, pi

# libraries for the plots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import meshgrid

# global parameters
surfaceplot_z_axis_limits = (-150, 150)
population_size = 60
elite_population_size = 10
x_max = 100
y_max = 100
mutation_probability = 0.05
mutation_range = 5
num_iterations = 15

# objective function to maximise
def objective_function(x, y):
    return (10 * sin((5 * pi * x)/(2 * x_max)) ** 2) * (10 * sin((5 * pi * y)/(2 * y_max)) ** 2) * exp(-(x + y)/100)
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
    # create random 2D point within boundaries [0, x_max] and [0, y_max]
    def __init__(self):
        self.x = uniform(0, x_max)
        self.y = uniform(0, y_max)
        self.index = -1
        self.fitness = 0.0
        self.diversity = 0.0
        self.fitness_rank = -1
        self.diversity_rank = -1

    # fitness score - objective function evaluated at the point
    def evaluate_fitness(self, eval_function=objective_function):
        return eval_function(self.x, self.y)

    # mutation
    def mutate(self):
        index = randint(0, 1)
        if index == 0:
            self.x += uniform(-mutation_range, mutation_range)

            # point shouldn't mutate out of range!
            self.x = min(self.x, x_max)
            self.x = max(self.x, 0)
        else:
            self.y += uniform(-mutation_range, mutation_range)

            # point shouldn't mutate out of range!
            self.y = min(self.y, y_max)
            self.y = max(self.y, 0)

        self.fitness = self.evaluate_fitness()


# Population class and method definition
class population2D:
    def __init__(self, size):
        self.points = []

        for pointnumber in range(size):
            point = Point2D()
            self.points.append(point)
            self.points[pointnumber].index = pointnumber

        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False
        self.mean_fitness = 0
        self.mean_diversity = 0
        self.x_mean = 0
        self.y_mean = 0

    # evaluate fitness rank of each point in population
    def evaluate_fitness_ranks(self):
        if not self.evaluated_fitness_ranks:
            for point in self.points:
                point.fitness = point.evaluate_fitness()
                self.mean_fitness += point.fitness

            self.mean_fitness /= population_size
            self.points.sort(key=lambda point: point.fitness, reverse=True)

            for rank_number in range(population_size):
                self.points[rank_number].fitness_rank = rank_number

            self.evaluated_fitness_ranks = True

    # evaluate diversity rank of each point in population
    def evaluate_diversity_ranks(self):
        if not self.evaluated_diversity_ranks:
            # find mean x and y coordinates
            self.x_mean = 0
            self.y_mean = 0

            for point in self.points:
                self.x_mean += point.x
                self.y_mean += point.y

            self.x_mean /= population_size
            self.y_mean /= population_size

            for point in self.points:
                point.diversity = (abs(point.x - self.x_mean) + abs(point.y - self.y_mean))
                self.mean_diversity += point.diversity

            self.mean_diversity /= population_size
            self.points.sort(key=lambda point: point.diversity, reverse=True)

            for rank_number in range(population_size):
                self.points[rank_number].diversity_rank = rank_number

            self.evaluated_diversity_ranks = True

    # generate the new population by breeding points
    def breed(self):
        # sort according to diversity and fitness rank
        self.points.sort(key=lambda point: point.fitness_rank)

        # push all the really good points first
        newpopulation = []
        for pointnumber in range(elite_population_size):
            newpopulation.append(self.points[pointnumber])

        # assign weights to being selected for breeding
        weights = [1 / (1 + point.fitness_rank + point.diversity_rank) for point in self.points]

        # randomly select for the rest and breed
        while len(newpopulation) < population_size:
            parent1 = weighted_choice(list(range(population_size)), weights)
            parent2 = weighted_choice(list(range(population_size)), weights)

            # don't breed with yourself, dude!
            while parent1 == parent2:
                parent1 = weighted_choice(list(range(population_size)), weights)
                parent2 = weighted_choice(list(range(population_size)), weights)

            # breed now
            child1, child2 = crossover(self.points[parent1], self.points[parent2])

            # add the children
            newpopulation.append(child1)
            newpopulation.append(child2)

        # assign the new population
        self.points = newpopulation
        self.evaluated_fitness_ranks = False
        self.evaluated_diversity_ranks = False

    # mutate population randomly
    def mutate(self):
        for point in self.points:
            test_probability = uniform(0, 1)
            if test_probability < mutation_probability:
                point.mutate()

                self.evaluated_fitness_ranks = False
                self.evaluated_diversity_ranks = False


# crossover (breed) 2 points by swapping x-coordinates
def crossover(point1, point2):
    child1 = Point2D()
    child2 = Point2D()

    child1.x = point1.x
    child1.y = point2.y

    child2.x = point2.x
    child2.y = point1.y

    child1.fitness = child1.evaluate_fitness()
    child2.fitness = child2.evaluate_fitness()

    return child1, child2

# surface plot of objective function - comment out if not interested
fig = plt.figure()
axes = fig.gca(projection='3d')

x_sample = range(x_max + 1)
y_sample = range(y_max + 1)
x_sample, y_sample = meshgrid(x_sample, y_sample)
z_sample = [[objective_function(x_sample[i][j], y_sample[i][j]) for i in range(x_max + 1)] for j in range(y_max + 1)]

surface = axes.plot_surface(x_sample, y_sample, z_sample, cmap=cm.coolwarm, linewidth=0, antialiased=False)

axes.set_zlim(surfaceplot_z_axis_limits)
axes.zaxis.set_major_locator(LinearLocator(10))
axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# color bar to map values to colours
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.title("Objective Function Surface Plot")
plt.show()


# actual algorithm code starts here #
# initial population
population = population2D(population_size)
population.evaluate_fitness_ranks()
population.evaluate_diversity_ranks()

# for plotting
plot_fitness = [population.mean_fitness]
plot_diversity = [population.mean_diversity]
population_x_vals = [point.x for point in population.points]
population_y_vals = [point.y for point in population.points]

plt.scatter(population_x_vals, population_y_vals)
plt.title("Initial Population")
plt.xlim((0, x_max))
plt.ylim((0, y_max))
plt.show()

# print the initial stats
print("Initial Population")
print("Mean fitness =", population.mean_fitness)
print("Mean L1 diversity =", population.mean_diversity)
print()

# breed and mutate this num_iterations times
for iteration in range(1, num_iterations + 1):

    population.breed()
    population.mutate()

    # find the new ranks
    population.evaluate_fitness_ranks()
    population.evaluate_diversity_ranks()

    # add to the plot
    plot_fitness.append(population.mean_fitness)
    plot_diversity.append(population.mean_diversity)

    # print the updated stats
    print("Iteration", iteration)
    print("Mean fitness =", population.mean_fitness)
    print("Mean L1 diversity =", population.mean_diversity)
    print()

# point with best fitness is the estimate of point of maxima
best_point = Point2D()
for point in population.points:
    if point.fitness > best_point.fitness:
        best_point = point

print("Function Maximum Estimate =", best_point.fitness)
print("Function Maximum Position Estimate =", "(" + str(best_point.x) + ", " + str(best_point.y) + ")")

# plotting again
# plot final population points
population_x_vals = [point.x for point in population.points]
population_y_vals = [point.y for point in population.points]

plt.scatter(population_x_vals, population_y_vals, color='r')
plt.title("Final Population")
plt.xlim((0, x_max))
plt.ylim((0, y_max))
plt.show()

# plot population stats
plt.subplot(2, 1, 1)
plt.title("Mean Population Stats")

plt.plot(range(num_iterations + 1), plot_fitness, 'b--',)
plt.ylabel('Fitness')

plt.subplot(2, 1, 2)
plt.plot(range(num_iterations + 1), plot_diversity, 'r--')
plt.ylabel('L1 diversity')

plt.xlabel("Iteration number")
plt.show()
