# Optimzer Visualisation

Python code that shows the effects of using different techniques for stochastic gradient descent. 

    pip install bayesian-optimization

This is a constrained global optimization package built upon bayesian inference
and gaussian process, that attempts to find the maximum value of an unknown
function in as few iterations as possible. This technique is particularly
suited for optimization of high cost functions, situations where the balance
between exploration and exploitation is important.

## Quick Start
In the [examples](https://github.com/fmfn/BayesianOptimization/tree/master/examples)
folder you can get a grip of how the method and this package work by:
- Checking out this
[notebook](https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb)
with a step by step visualization of how this method works.
- Going over this
[script](https://github.com/fmfn/BayesianOptimization/blob/master/examples/usage.py)
to become familiar with this packages basic functionalities.
- Exploring this [notebook](https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb)
exemplifying the balance between exploration and exploitation and how to
control it.
- Checking out these scripts ([sklearn](https://github.com/fmfn/BayesianOptimization/blob/master/examples/sklearn_example.py),
[xgboost](https://github.com/fmfn/BayesianOptimization/blob/master/examples/xgboost_example.py))
for examples of how to use this package to tune parameters of ML estimators
using cross validation and bayesian optimization.
## Setup

We are going to use stochastic gradient descent to learn the values of a and b in this function:

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/maineqn.gif)

which looks like: 

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/func.png)

## Comparing Optimizers

Vanilla SGD can't deal with how noisy the information it's getting is, and fails to find the global optimum completely! The black lines below show true
values for a and b, and the red dot is where we started from (a = 0.8, b = 0.9)

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/sgd_5.png)

Since Adam uses momentum it can deal much better with the noisy information given by the sampled x values. 

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/adam_5.png)

As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account what it knows about the target function. At each step a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with a exploration strategy (such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that should be explored (see the gif below).

![BayesianOptimization in action](https://github.com/fmfn/BayesianOptimization/blob/master/examples/bayesian_optimization.gif)

This process is designed to minimize the number of steps required to find a combination of parameters that are close to the optimal combination. To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. Therefore Bayesian Optimization is most adequate for situations where sampling the function to be optimized is a very expensive endeavor. See the references for a proper discussion of this method.

This project is under active development, if you find a bug, or anything that
needs correction, please let me know.

Installation
============

### Installation

For the latest release, run:

    pip install bayesian-optimization

The bleeding edge version can be installed with:

    pip install git+https://github.com/fmfn/BayesianOptimization.git

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies:

    git clone https://github.com/fmfn/BayesianOptimization.git
    cd BayesianOptimization
    python setup.py install

### Dependencies
* Numpy
* Scipy
* Scikit-learn

### References:
* http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
* http://arxiv.org/pdf/1012.2599v1.pdf
* http://www.gaussianprocess.org/gpml/
* https://www.youtube.com/watch?v=vz3D36VXefI&index=10&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6