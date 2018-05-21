## fmga
**fmga** (**f**unction **m**aximization through **g**enetic **a**lgorithms) is a package that takes a genetic algorithm approach to maximization problem of non-convex objective functions in two dimensions.
 
The objective function doesn't have to be differentiable, or even continuous in the specified domain!  

The idea is to sample an evolving population of points converging to the function maximum over many iterations.

The population of 2-dimensional points undergoes random mutations - and is selected through elitism along with breeding with selection weights inversely proportional to fitness and diversity ranks.


### Installation
Install with pip:
```bash
pip install fmga
```
Import within the python script with:
```python
import fmga
```

### Execution
Given a function on two variables x and y, say:
```python
def f(x, y):
    return x - math.sin(y) + 28
```
Pass this function as the *objective_function* argument to the **population2D** constructor (lambdas work too!):
```python
population = fmga.population2D(objective_function=f, population_size=60)
```
The population can be set to breed and iterate by using the **.converge()** method.
```python
population.converge(iterations=20)
```
To perform only one iteration of breeding and mutating, do:
```python
population.iterate()
```
Access population mean fitness and mean L1 diversity stats through the _.mean_fitness_ and _.mean_diversity_ attributes:
```python
print(population.mean_fitness, population.mean_diversity)
```

The **.best_estimate()** method returns the point closest to the function point of maxima in the population, as a **Point2D** object.
```python
best_point = population.best_estimate()
```
Every **Point2D** object has attributes 'x' and 'y', signifying the coordinates of the maxima point.
```python
print(best_point.x, " ", best_point.y)
```
To find the value of the function at this point, use:
```python
print(best_point.fitness)
```

## population2D Class Methods
The population2D constructor takes the following arguments:

**population_size** (default = 60) Number of points in the population.  
**objective_function** The function to maximize!  
**elite_fraction** (default = 0.1) Fraction of the population's points to be kept as elite during breeding. Must be between 0 and 1, inclusive.  
**x_min, x_max, y_min, y_max** (default = 0, 100, 0, and 100 respectively) The domain where the points are spread.  
**mutation_probability** (default = 0.05) How likely is is for a single point to mutate - this probability is the same for all points in the population.
Must be between 0 and 1, inclusive.  
**mutation_range** (default = 5) The range of the mutation when it does occur. Note that the point will never mutate out of the domain defined!  
**verbose** (default = 2) How much output to be displayed when iterating population after population. Must take values 0, 1 or 2 with 2 representing the most output, and 0 representing none.

