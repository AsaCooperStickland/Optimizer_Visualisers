# Optimzer Visualisation

Python code that shows the effects of using different techniques for stochastic gradient descent. 


## Setup

We are going to use stochastic gradient descent to learn the values of a and b in this function:

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/maineqn.gif)

which looks like: 

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/func.png)

## Comparing Optimizers

Vanilla SGD can't deal with how noisy the information it's getting is, and fails to find the global optimum completely! The black lines below show true
values for a and b, and the red dot is where we started from (a = 0.8, b = 0.9)

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/sgd_5.png)

Since Adam uses momentum it can deal much better with the noisy information given by the sampled x values. the random x values are providing us with
gradients that on average point in the right direction. Since momentum gives us a moving average of the gradient the Adam method can find it's way
to the optimum. Note that it does occasionally fail- that's the path going to the top right, which has found a local minima, but not the global one. 
This can be stopped by setting the learning rate to a lower value, which would of course mean it takes longer to find the optimum. 

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/adam_5.png)

Adam also uses a moving average of the variance of the gradient, this is used to boost the learning rate of variables that are 'rarer'. But that 
doesn't really apply here since we only have two variables for visualisation purposes. 


### Dependencies
* Numpy
* Matplotlib
### References:
* http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
* http://arxiv.org/pdf/1012.2599v1.pdf
* http://www.gaussianprocess.org/gpml/
* https://www.youtube.com/watch?v=vz3D36VXefI&index=10&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6