# GeneticAlgorithmsRepo

### Objective Function Maximization
**function_maximize.py** takes an objective function of two variables and samples an evolving population of points converging to the function maxima.

The population undergoes random mutations - and is selected through elitism along with breeding with selection weights 
inversely proportional to fitness and diversity ranks.  

The objective function doesn't have to be differentiable, or even continuous in the specified domain!
Included are surface plots of the function, initial population and final population graphs, as well as line plots of mean population fitness 
and L1 diversity through the iterations.
Completely customizable and free to use.

Runs in Python 3.5.


