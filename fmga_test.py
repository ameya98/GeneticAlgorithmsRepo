'''
Testing the fmga package
Author: Ameya Daigavane
'''

from fmga import maximize
import math


if __name__ == '__main__':

    def f(x, y, z):
        return -z * math.sin(x + y) + z * math.sin(x - y)

    bounds = [(-math.pi, math.pi), (-math.pi, math.pi), (-10, 10)]
    best_point = maximize(f, dimensions=3, mutation_probability=0.1, population_size=100, multiprocessing=True, iterations=10, boundaries=bounds)
    print(best_point.coordinates, best_point.fitness)

