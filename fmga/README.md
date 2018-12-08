## fmga
**fmga** (**f**unction **m**aximization through **g**enetic **a**lgorithms) is a package that takes a genetic algorithm approach to maximization problem of non-convex objective functions in multiple dimensions.
 
The objective function doesn't have to be differentiable, or even continuous in the specified domain!  
The idea is to sample an evolving population of points converging to the function maximum over many iterations.

The population of n-dimensional points undergoes random mutations - and is selected through elitism and ranking selection with selection weights inversely proportional to fitness and diversity ranks.

**fmga** now supports multiprocessing through **[pathos](https://github.com/uqfoundation/pathos)** too! 

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
Given a function on multiple variables, say:
```python
def f(x, y, z):
    return x - math.sin(y) * z
```
Pass this function as the *objective_function* argument to the .maximize() method (lambdas work too!).  

```python
best_point = fmga.maximize(f, population_size=60, dimensions=3)
```

The **maximize()** method creates a **Population** of **Point** objects, calls the **.converge()** method on the Population object, and finally,
returns a **Point** object representing the n-dimensional point with best fitness through the **.best_estimate()** method.  

```python
print(best_point, best_point.fitness)
```
By default, the *multiprocessing* argument defaults to False, so to enable multiprocessing, set this argument to True, and pass the number of processes to be spawned as the *processes* argument.
```python
best_point = fmga.maximize(f, multiprocessing=True, processes=4)
```
Note that, when multiprocessing is enabled on Windows systems, you must put a guard over the entry point of your script.
See [here](https://docs.python.org/2/library/multiprocessing.html#windows) for a how-to.

fmga also supports a variable number of dimensions to optimise over, passed as the *dimensions* argument, which defaults to the number of arguments of the objective function passed.

If you wish to interact with the **Population** object directly, you can.
Both of the following work:
```python
population = fmga.Population(f, population_size=60, dimensions=3)
population = fmga.Population(population_size=60, objective_function=f, dimensions=3)
```
If you wish to define custom boundaries, create a list of tuples, for each dimension. Default boundaries are (0, 100). 
(This is different than in versions 1.x)
```python
boundaries = [(0, 2.5), (0, 10)]
```
and pass this as the *boundaries* argument to the **Population** constructor or the **maximise()** method:
```python
population = fmga.Population(f, population_size=60, boundaries=boundaries)
best_point = fmga.maximize(f, population_size=60, boundaries=boundaries)
```
Note that the default range for missing dimensions is (0, 100).  
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

The **.best_estimate()** method returns the point closest to the function point of maxima in the population, as a **Point** object.
```python
best_point = population.best_estimate()
```
Every **Point** object has the __coordinates__ attribute, a numpy array signifying the coordinates of point.
To find the value of the function at this point, use the __fitness__ attribute.
```python
print(best_point.coordinates)
print(best_point.fitness)
```

## Population Class Methods
The Population constructor takes the following arguments, in order:  
**objective_function** The function to maximize!  
**dimensions** (default = number of arguments of objective_function) The dimensionality of the points and the number of variables to maximize over.  

From versions 2.8.0 and onward, the PopulationParameters class handles the parameters below. 
The interface is the same as previous versions, however, so you can pass these arguments to the Population constructor as before.

**population_size** (default = 60) Number of points in the population.  
**boundaries** (default = (0, 100) for every dimension) Must be an iterable of tuples. The tuple indicates the domain where the points are spread along that dimension.    
**elite_fraction** (default = 0.1) Fraction of the population's points to be kept as elite during breeding. Must be between 0 and 1, inclusive.  
**mutation_probability** (default = 0.05) How likely is is for a single point to mutate - this probability is the same for all points in the population.
Must be between 0 and 1, inclusive.  
**mutation_range** (default = 5) The range of the mutation when it does occur. Note that the point will never mutate out of the domain defined!     
**multiprocessing** (default = False) Whether multiprocessing is enabled  
**processes** (default = multiprocessing.cpu_count()) Number of processes to spawn if multiprocessing is enabled. 

The **maximize()** method takes all of the above, an **iterations** argument,
defaulting to 15, signifying the number of iterations that the underlying population undergoes, as well as a **verbose** argument (default = 0, was 2 for versions <= 2.4.0) denoting how much console output to be displayed after each iteration (Must take values 0, 1 or 2 with 2 representing the most output, and 0 representing none.)

The **converge()** and **iterate()** methods also take the **iterations** and **verbose** arguments.

The **minimize()** method is a wrapper over the **maximize()** method - replacing the objective function by its negative, and maximizing this new objective function.

The **unpack()** method accepts two arguments, a tuple of values and a list of shapes. If the length of the list of shapes is greater than one, it returns a list of numpy arrays of shape according to the list, by reshaping the tuple in-order.
Otherwise it returns just a numpy array of the passed shape, formed by reshaping the tuple.   
This is useful when working with a large number of arguments! Example:
```python
def f(*args):
    x, y, z = fmga.unpack(args, (1, (2, 2), 4))
    
    # x.shape == (1,)
    # y.shape == (2, 2)
    # z.shape == (4,)
    
    return x - y[0][0] + z[2]
```

## Dependencies
* numpy
* pathos (>= 0.2.2.1)