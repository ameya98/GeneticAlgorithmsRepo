# Testing the fmga package#
# Author: Ameya Daigavane #
# Date: 15th April, 2018 #

# libraries for the genetic algorithm
from fmga import population2D
from math import sin, pi, exp

# libraries for the plots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# objective function to maximise
def f(x, y):
    return (10 * sin((5 * pi * x)/(2 * 100)) ** 2) * (10 * sin((5 * pi * y)/(2 * 100)) ** 2) * exp(-(x + y)/100)


# surface plot of objective function - comment out if not interested
fig = plt.figure()
axes = fig.gca(projection='3d')

x_sample = np.arange(101)
y_sample = np.arange(101)
x_sample, y_sample = np.meshgrid(x_sample, y_sample)
z_sample = [[f(x_sample[i][j], y_sample[i][j]) for i in range(101)] for j in range(101)]

surface = axes.plot_surface(x_sample, y_sample, z_sample, cmap=cm.coolwarm, linewidth=0, antialiased=False)

axes.set_zlim(-50, 50)
axes.zaxis.set_major_locator(LinearLocator(10))
axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# color bar to map values to colours
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.title("Objective Function Surface Plot")
plt.show()


# actual algorithm code starts here #
# initial population
population = population2D(population_size=60, objective_function=f)

# for plotting
plot_fitness = [population.mean_fitness]
plot_diversity = [population.mean_diversity]
population_x_vals = [point.x for point in population.points]
population_y_vals = [point.y for point in population.points]

plt.scatter(population_x_vals, population_y_vals)
plt.title("Initial Population")
plt.xlim((0, 100))
plt.ylim((0, 100))
plt.show()

# print the initial stats
print("Initial Population")
print("Mean fitness =", population.mean_fitness)
print("Mean L1 diversity =", population.mean_diversity)
print()

# breed and mutate this num_iterations times
for iteration in range(1, 16):

    # perform one iteration
    population.iterate()

    # add to the plot after every iteration
    plot_fitness.append(population.mean_fitness)
    plot_diversity.append(population.mean_diversity)

    # print the updated stats
    print("Iteration", iteration)
    print("Mean fitness =", population.mean_fitness)
    print("Mean L1 diversity =", population.mean_diversity)
    print()

# point with best fitness is the estimate of point of maxima
best_point = population.best_estimate()

print("Function Maximum Estimate =", best_point.fitness)
print("Function Maximum Position Estimate =", "(" + str(best_point.x) + ", " + str(best_point.y) + ")")

# plotting again
# plot final population points
population_x_vals = [point.x for point in population.points]
population_y_vals = [point.y for point in population.points]

plt.scatter(population_x_vals, population_y_vals, color='r')
plt.title("Final Population")
plt.xlim((0, 100))
plt.ylim((0, 100))
plt.show()

# plot population stats
plt.subplot(2, 1, 1)
plt.title("Mean Population Stats")

plt.plot(range(16), plot_fitness, 'b--',)
plt.ylabel('Fitness')

plt.subplot(2, 1, 2)
plt.plot(range(16), plot_diversity, 'r--')
plt.ylabel('L1 diversity')

plt.xlabel("Iteration number")
plt.show()
