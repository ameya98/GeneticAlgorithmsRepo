import fmga

def f(*args):
    return args[0] - args[1] + args[2]

best_point = fmga.maximize(f, dimensions=3, mutation_probability=0.1, population_size=1000)
print(best_point.coordinates, best_point.fitness)

