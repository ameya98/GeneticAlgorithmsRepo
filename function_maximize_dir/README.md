## function_maximize_genetic
**function_maximize_genetic** is a package that solves the problem of maximizing non-convex objective functions in 2-dimensions,
with a genetic algorithm approach.
The idea is to sample an evolving population of points converging to the function maximum.

The population is interfaced with the **population2D** object:
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

Remember to define the function having two variables x and y as input, just like:
```python
def f(x, y):
    return (x - y + 28)
```

The objective function doesn't have to be differentiable, or even continuous in the specified domain!
The population of 2-dimensional points undergoes random mutations - and is selected through elitism along with breeding with selection weights inversely proportional to fitness and diversity ranks.