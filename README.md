# Optimzer Visualisation

Python code that shows the effects of using different techniques for stochastic gradient descent. 


## Setup

We are going to use stochastic gradient descent to learn the values of a (0.25) and b (0.75) in this function:

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/maineqn.gif)

which looks like: 

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/func.png)

Say we have 100 points between 0 and 1, and work out the error between the true function (with a = 0.25 and b = 0.75) and our guess for what a and b are.
When we map out the errors it looks like this:

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/true_error.png)

There's obviously a minimum at a = 0.25, b = 0.75 where the error goes to zero, and it looks like a = 0.8, b = 1.0 is not a terrible fit. What if we don't 
have 100 points, what if we just have 2? 

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/err_surf_2samples.png)

These are error surfaces with just two randomly sampled x values. They look similiar-ish, but some of them are missing maxima or minima. This is all our 
optimzer has access to at each time step. If you sampled enough times and took an average it would look like the true error above- this is where 'momentum' 
comes in. If we have some way of storing the average direction we're going in it's going to help us find the minima. 


## Comparing Optimizers

Vanilla SGD can't deal with how noisy the information it's getting is, and fails to find the global optimum completely! The black lines below show true
values for a and b, and the red dot is where we started from (a = 0.8, b = 0.9)

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/sgd_5.png)

Since Adam uses momentum it can deal much better with the noisy information given by the sampled x values. the random x values are providing us with
gradients that on average point in the right direction. Since momentum gives us a moving average of the gradient the Adam method can find it's way
to the optimum. 

![Five Adam runs](https://github.com/AsaCooperStickland/Optimizer_Visualisers/blob/master/images/adam_5.png)

Adam also uses a moving average of the variance of the gradient, this is used to boost the learning rate of variables that are 'rarer' by dividing the learning
rate by the standard deviation of the gradient. But that doesn't really apply here since we only have two variables (for visualisation purposes). It is of course very useful in 
deep learning tasks where we might have millions (or if we have access to fancy GPUs, billions!) of variables. 

I've also included the 'Eve' optimizer that is an extension to Adam that basically increases the learning rate when the 
objective is changning slowly, and decreases it when it changes rapidly. The method isn't invariant to adding a constant 
term to the objective, and it's not clear that it's worth the extra hyperparameters it introduces (and the paper got rejected
from ICLR 2017 so...). So I won't go into it too much here but it's in the code if you want to play around with it!

### References:
* http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
* http://arxiv.org/pdf/1012.2599v1.pdf
* http://www.gaussianprocess.org/gpml/
* https://www.youtube.com/watch?v=vz3D36VXefI&index=10&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6