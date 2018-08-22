## testing the global package
import fmga
import math


if __name__ == '__main__':

    def f(*args):
        sum = args[0] - args[1] + args[2]
        for num in range(100000):
            sum += math.sin(num)
        return sum

    best_point = maximize(f, dimensions=3, mutation_probability=0.1, population_size=100, multiprocessing=True, iterations=5)
    print(best_point.coordinates, best_point.fitness)

