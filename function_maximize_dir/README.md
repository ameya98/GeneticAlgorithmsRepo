## function_maximize_genetic
**function_maximize_genetic** is a package that solves the problem of maximizing non-convex objective functions in 2-dimensions,
with a genetic algorithm approach.  
The idea is to sample an evolving population of points converging to the function maximum.

Given a function on two variables x and y, say:
```python
def f(x, y):
    return (x - y + 28)
```
Pass this function as the *objective_function* argument to the **population2D** constructor:
```python
population = population2D(objective_function=f, population_size=60, elite_fraction=0.05)
```
and to find the maximum, just do:
```python
best_point = population.converge(iterations=20)
```
**population.converge()** returns a **Point2D** object, which has attributes 'x' and 'y', signifying the coordinates.
```python
print(best_point.x, " ", best_point.y)
```
To find the value of the function at this point, use:
```python
print(best_point.fitness)
```

The objective function doesn't have to be differentiable, or even continuous in the specified domain!  
The population of 2-dimensional points undergoes random mutations - and is selected through elitism along with breeding with selection weights inversely proportional to fitness and diversity ranks.

## population2D Class Methods
The population2D constructor takes the following arguments:
#### population_size
(default = 60) Number of points in the population.
#### objective_function
#### elite_fraction
(default = 0.1)
Fraction of the population's points to be kept as elite during breeding.
#### x_min, x_max, y_min, y_max
(default = 0, 100, 0, 100 respectively)
The domain where the points are spread.
#### mutation_probability
(default = 0.05)
How likely is is for a single point to mutate - this probability is the same for all points in the population.
#### mutation_range
(default = 5)
The range of the mutation when it does occur. Note that the point will never mutate out of the domain defined!
#### verbose
(default = 2)
How much output to be displayed when iterating population after population. Can take values 0, 1 or 2 - 2 representing the most output, and 0 representing none.

