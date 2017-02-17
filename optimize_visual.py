from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import matrix
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['font.family'] = 'serif'
label_size = 19
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
import os

class Optimize(object):


    """
    Comparing various optimizers for stochastic gradient descent on a very
    simple toy problem.

    Adam paper: https://arxiv.org/abs/1412.6980
    Eve paper: https://arxiv.org/pdf/1611.01505.pdf
    """

    
    def __init__(self, a, b, init, x_data, stochastic = True, samples = 2):


        """
        Initialize variables

        :param a:
            True value of a.
        :param b:
            True value of b.
        :param init:
            Array with intial approximate a and b.
        :param x_data:
            Set of points where the true and approximate functions can be
            compared
        :param stochastic:
            Bool, if true sample x values randomly, if false just use all the
            data.
        :param samples:
            If sampling x values, this is how many to choose.
        """

        
        self.a_true = a
        self.b_true = b
        self.a_approx = init[0]
        self.a_rec = np.array([])
        self.b_approx = init[1]
        self.b_rec = np.array([])
        self.x_data = x_data
        self.stochastic = stochastic
        self.samples = samples
        if stochastic == True:
            self.get_minibatch()
        else:
            self.xs = self.x_data
        # Place to store variables in Adam and Eve optimizers. 
        self.m_old = np.zeros(2)
        self.m = np.zeros(2)
        self.v_old = np.zeros(2)
        self.v = np.zeros(2)
        self.d = 1.0
        self.d_old = 1.0
        self.f_store = np.zeros(2) + self.err(self.xs)
        self.grad = self.calc_grad()

    def get_minibatch(self):

        """
        Gets random sample from x values of size 'samples'.
        """
        
        self.xs = np.random.choice(self.x_data, size=self.samples)
        
    def true_func(self, a, b, x):

        """
        Function we're estimating values of.
        """
        
        return ((1.0/0.05**0.5)*np.exp(-(x - a)**2/0.05) +
            (1.0/0.01**0.5)*np.exp(-(x - b)**2/0.01))

    def err(self, x):

        """
        Returns mean error between our current guess for a and b and their true
        values.
        """
        
        tot = (self.true_func(self.a_true, self.b_true, x) -
               self.true_func(self.a_approx, self.b_approx, x))**2
        return np.mean(tot)

    def err_dash_simp(self, x):

        """
        First step in chain rule for derivative of error.
        """
        
        tot = 2.0*(self.true_func(self.a_true, self.b_true, x) -
               self.true_func(self.a_approx, self.b_approx, x))
        return tot

    def err_dash(self, x):

        """
        Derivative of error.
        """
        
        a_part = -((2.0/0.05)*(x - self.a_approx)*((1.0/0.05**0.5)*
                                      np.exp(-(x - self.a_approx)**2/0.05)))
        b_part = -((2.0/0.01)*(x - self.b_approx)*((1.0/0.01**0.5)*
                                      np.exp(-(x - self.b_approx)**2/0.01)))
        err_dash_simp = self.err_dash_simp(x)
        return np.array([np.mean(a_part*err_dash_simp),
                         np.mean(b_part*err_dash_simp)])

    def calc_grad(self):

        """
        Get random set of points, then calculate gradients for a and b.
        """
        
        if self.stochastic == True:
            self.get_minibatch()
        func_dash = self.err_dash(self.xs)
        self.grad = func_dash

    def stoch_grad_desc(self, eta):

        """
        Standard SGD update step.

        :param eta:
            learning rate, controls how much to change variables at each
            optimization step. 
        """
        
        self.a_approx = self.a_approx - eta*self.grad[0] 
        self.b_approx = self.b_approx - eta*self.grad[1]

    def adam(self, eta, beta_1, beta_2, epsilon, t):

        """
        Adam update step. See 'Adam: a method of stochastic optimization' for
        more details.

        eta is as before.

        :param beta_1:
            Momentum term: amount of old gradient to keep.

        :param beta_2:
            More momentum, this time for square of gradient (stand in for
            variance of gradient.).

        :param epsilon:
            Fudge factor to prevent division by zero.

        :param t:
            Current timestep. 
        """
        
        self.m = beta_1*self.m_old + (1 - beta_1)*self.grad
        self.v = beta_2*self.v_old + (1 - beta_2)*self.grad**2
        m_hat = self.m/(1.0 - beta_1**t)
        v_hat = self.v/(1.0 - beta_2**t)
        self.a_approx = self.a_approx - eta*m_hat[0]/(v_hat[0]**0.5 + epsilon) 
        self.b_approx = self.b_approx - eta*m_hat[1]/(v_hat[1]**0.5 + epsilon)
        self.m_old = self.m
        self.v_old = self.v

    def eve(self, eta, beta_1, beta_2, beta_3, little_k, big_k, epsilon, t, x):

        """
        Eve update step.

        Most parameters the same as in Adam.

        :param beta_3:
            Momentum term for the variability of the objective.

        :param little_k:
            Lower limit on how much to modify gradient

        :param big_k:
            Upper limit on same.

        :param x:
            Current sampled x values, for calculating objective. 
        """
        
        self.m = beta_1*self.m_old + (1 - beta_1)*self.grad
        self.v = beta_2*self.v_old + (1 - beta_2)*self.grad**2
        m_hat = self.m/(1.0 - beta_1**t)
        v_hat = self.v/(1.0 - beta_2**t)
        f = self.err(x)
        
        if t > 1:    
            if f > self.f_store[0]:
                delta_t = little_k + 1
                tri_t = big_k + 1
            else:
                delta_t = 1.0/big_k
                tri_t = 1.0/little_k
            c_t = min(max(delta_t, f/self.f_store[0]) , tri_t)
            self.f_store[1] = c_t*self.f_store[0]
            r_t = abs(self.f_store[1] - self.f_store[0])/min(self.f_store)
            
            if r_t < 1.0:
                self.d = beta_3*self.d_old + (1 - beta_3)*r_t
        else:
            self.f_store[1] = f
            self.d = 1.0
            
        self.a_approx = self.a_approx - eta*m_hat[0]/(self.d*v_hat[0]**0.5 +
                                                      epsilon) 
        self.b_approx = self.b_approx - eta*m_hat[1]/(self.d*v_hat[1]**0.5 +
                                                      epsilon)
        self.m_old = self.m
        self.v_old = self.v
        self.d_old = self.d
        self.f_store[0] = self.f_store[1]
        self.f_store[1] = f
        

    def opt(self, n, method = 'sgd'):

        """
        Run an optimisation method (e.g. Adam) for n steps, and record the
        values of a and b after each step.

        :param n:
            Number of steps to optimize.

        :param method:
            String that decided which optimization method to use.
        """
        
        for i in range(n):
            self.a_rec = np.append(self.a_rec, self.a_approx)
            self.b_rec = np.append(self.b_rec, self.b_approx)
            self.calc_grad()
            if method == 'sgd':
                self.stoch_grad_desc(0.00005)
            if method == 'adam':
                self.adam(0.02, 0.9, 0.999, 10e-8, i+1)
            if method == 'eve':
                self.eve(0.02, 0.9, 0.999, 0.999, 0.1, 10.0, 10e-8,
                         i+1, self.xs)

    def input_params(self, a_new, b_new):

        """
        Input new a and b values
        """
        
        self.a_approx = a_new
        self.b_approx = b_new

    def clear_rec(self):

        """
        Reset all stored variables
        """
        
        self.a_rec = np.array([])
        self.b_rec = np.array([])

        self.m_old = np.zeros(2)
        self.m = np.zeros(2)
        self.v_old = np.zeros(2)
        self.v = np.zeros(2)
        self.d = 1.0
        self.d_old = 1.0
        self.f_store = np.zeros(2) + self.err(self.xs)
        self.grad = self.calc_grad()

    def generate_paths(self, k, n, method = 'sgd'):

        """
        Generate k optimisation runs of n steps starting
        at (a = 0.8, b = 0.9), and record the results.

        :param k:
            Number of runs to do.

        :param n:
            Number of steps in each run.

        :pram method:
            String that decided whcih optimzation method to use. 
        """

        # Big arrays with as and bs for each run. 
        output_a = np.zeros([k, n])
        output_b = np.zeros([k, n])
        for i in range(k):
            self.input_params(0.8, 0.9)
            self.opt(n, method)
            output_a[i, :] = self.a_rec
            output_b[i, :] = self.b_rec
            self.clear_rec()
        return output_a, output_b
                      
def true_func(x, a, b):

    """
    Function we find values for
    """
    
    return ((1.0/0.05**0.5)*np.exp(-(x - a)**2/0.05) +
            (1.0/0.01**0.5)*np.exp(-(x - b)**2/0.01))
 
def err(xs, a_approx, b_approx):

    """
    Errror between appromation and real function
    """
    
    tot = (true_func(xs, a_approx, b_approx) - true_func(xs, 0.25, 0.75))**2
    return np.mean(tot)

def err_surface(x_data, a_guess, b_guess):

    """
    Generate error for a bunch of differnt values of a and b, from the arrays
    a_guess and b_guess
    """

    l_a = len(a_guess)
    l_b = len(b_guess)
    errors = np.zeros([l_a, l_b])
    for i in range(l_a):
        for j in range(l_b):
            errors[l_a - i - 1, j] = err(x_data, a_guess[i], b_guess[j])
    return errors

def plot_paths(methods = ['sgd', 'adam'], runs = 5, run_length = 50):

    """
    Plot on an error surface the paths that several runs of optimisations took

    :param methods:
        Array containing strings that decide which method to use.

    :param runs:
        Number of times optimization is run.

    :param run_length:
        Length of each run. 
    """

    x_data = np.arange(0, 1, 0.01)
    a_guess = np.arange(0.0, 1.0, 0.01)
    b_guess = np.arange(0.5, 1.5, 0.01)
    errors = err_surface(x_data, a_guess, b_guess)
    Example = Optimize(0.25, 0.75, [0.8, 0.9], x_data, stochastic = True)
    for method in methods:
        a_s, b_s = Example.generate_paths(runs, run_length, method)
        plt.figure()
        plt.imshow(errors, cmap = 'Blues', interpolation='bilinear',
               extent=[0.0,1.0,0.5,1.5], vmin=0, vmax=40)
        for i in range(runs):
            plt.plot(a_s[i,:], b_s[i,:], color = 'green')
            plt.scatter(a_s[i,:], b_s[i,:], s= 30, alpha=0.3,
                    edgecolor='black', facecolor='g', linewidth=0.75)
        plt.scatter(0.8, 0.9, s= 30, alpha=1.0, edgecolor='black',
                facecolor='r', linewidth=0.75)
        plt.plot((0.25, 0.25), (0.5, 1.2), 'k-')
        plt.plot((0.0, 1.0), (0.75, 0.75), 'k-')
        plt.ylabel('$b$', fontsize = 20)
        plt.xlabel('$a$', fontsize = 20)
        plt.ylim(0.5, 1.2)
        plt.xlim(0.0, 1.0)
        save_dir = 'images'
        plt.savefig(os.path.join(save_dir, method + "_" + str(runs) +
                             "runs_" + str(run_length) + "length.png"))
        plt.show()

def plot_stochastic_surface(x_data, samples):

    """
    Plots true error surface and then four error surfaces only using
    a certain number of data points, given by 'samples'.
    """
    
    a_guess = np.arange(0.0, 1.0, 0.01)
    b_guess = np.arange(0.5, 1.5, 0.01)
    true_err = err_surface(x_data, a_guess, b_guess)

    plt.figure()
    plt.imshow(true_err, cmap = 'Blues', interpolation='bilinear',
               extent=[0.0,1.0,0.5,1.5], vmin=0, vmax=40)
    plt.plot((0.25, 0.25), (0.5, 1.5), 'k-')
    plt.plot((0.0, 1.0), (0.75, 0.75), 'k-')
    plt.ylabel('$b$', fontsize = 20)
    plt.xlabel('$a$', fontsize = 20)
    plt.ylim(0.5, 1.2)
    plt.xlim(0.0, 1.0)
    plt.colorbar()
    save_dir = 'images'
    plt.savefig(os.path.join(save_dir, "true_error.png"))
    plt.show()

    errors = np.zeros([4, len(a_guess), len(b_guess)])
    for i in range(4):
        x = np.random.choice(x_data, size=samples)
        errors[i, :, :] = err_surface(x, a_guess, b_guess)
        
    plt.figure()
    f, axes = plt.subplots(2, 2, sharex='col', sharey='row')
    for err, ax, i in zip(errors, axes.flat, np.arange(0, 4)):
        im = ax.imshow(err, cmap = 'Blues', interpolation='bilinear',
               extent=[0.0,1.0,0.5,1.2], vmin=0, vmax=40)
        if i == 0 or i == 2:
            ax.set_ylabel('$b$', fontsize = 20)
        if i == 2 or i == 3:
            ax.set_xlabel('$a$', fontsize = 20)
        ax.plot((0.25, 0.25), (0.5, 1.2), 'k-')
        ax.plot((0.0, 1.0), (0.75, 0.75), 'k-')
        ax.set_ylim(0.5, 1.2)
        ax.set_xlim(0.0, 1.0)
    plt.tight_layout()
    save_dir = 'images'
    plt.savefig(os.path.join(save_dir, "err_surf_" + str(samples) +
                             "samples.png"))
    plt.show()

if __name__ == "__main__":
    x_data = np.arange(0, 1, 0.01)
    figs = plot_stochastic_surface(x_data, 2)
    figs = plot_paths(methods = ['adam', 'sgd'])
    

        







