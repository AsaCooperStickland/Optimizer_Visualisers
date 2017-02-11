from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import matrix
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['font.family'] = 'serif'
label_size = 19
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
'''from numpy import genfromtxt
import scipy.stats
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
from scipy.stats import invgamma
from scipy.stats import beta
from datetime import datetime'''
import os

class Optimize(object):
    def __init__(self, a, b, init, x_data, stochastic = True, samples = 2):
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
        self.m_old = np.zeros(2)
        self.m = np.zeros(2)
        self.v_old = np.zeros(2)
        self.v = np.zeros(2)
        self.grad = self.calc_grad()

    def get_minibatch(self):
        self.xs = np.random.choice(self.x_data, size=self.samples)
        #print(self.xs)
        
    def true_func(self, a, b, x):
        return ((1.0/0.05**0.5)*np.exp(-(x - a)**2/0.05) +
            (1.0/0.01**0.5)*np.exp(-(x - b)**2/0.01))

    def true_func_dash(self, x):
        a_part = -((2.0/0.05)*(x - self.a_approx)*((1.0/0.05**0.5)*
                                      np.exp(-(x - self.a_approx)**2/0.05)))
        b_part = -((2.0/0.01)*(x - self.b_approx)*((1.0/0.01**0.5)*
                                      np.exp(-(x - self.b_approx)**2/0.01)))
        err_dash = self.err_dash(x)
        return np.array([np.mean(a_part*err_dash), np.mean(b_part*err_dash)])

    def err(self, x):
        tot = (self.true_func(self.a_true, self.b_true, x) -
               self.true_func(self.a_approx, self.b_approx, x))**2
        return np.mean(tot)

    def err_dash(self, x):
        tot = 2.0*(self.true_func(self.a_true, self.b_true, x) -
               self.true_func(self.a_approx, self.b_approx, x))
        return tot

    def calc_grad(self):
        if self.stochastic == True:
            self.get_minibatch()
        func_dash = self.true_func_dash(self.xs)
        self.grad = func_dash

    def stoch_grad_desc(self, eta):
        #self.grad()
        self.a_approx = self.a_approx - eta*self.grad[0] 
        self.b_approx = self.b_approx - eta*self.grad[1]
        #print(eta*self.grad)

    def adam(self, eta, beta_1, beta_2, epsilon, t):
        self.m = beta_1*self.m_old + (1 - beta_1)*self.grad
        self.v = beta_2*self.v_old + (1 - beta_2)*self.grad**2
        m_hat = self.m/(1.0 - beta_1**t)
        v_hat = self.v/(1.0 - beta_2**t)
        #print(m_hat)
        self.a_approx = self.a_approx - eta*m_hat[0]/(v_hat[0]**0.5 + epsilon) 
        self.b_approx = self.b_approx - eta*m_hat[1]/(v_hat[1]**0.5 + epsilon)
        #print((v_hat[1]**0.5 + epsilon))
        self.m_old = self.m
        self.v_old = self.v
        #print('a', eta*m_hat[0]/(v_hat[0]**0.5 + epsilon))

    def opt(self, n, method = 'sgd'):
        for i in range(n):
            self.a_rec = np.append(self.a_rec, self.a_approx)
            self.b_rec = np.append(self.b_rec, self.b_approx)
            self.calc_grad()
            if method == 'sgd':
                self.stoch_grad_desc(0.00005)
            if method == 'adam':
                self.adam(0.03, 0.9, 0.999, 10e-8, n)

    def input_params(self, a_new, b_new):
        self.a_approx = a_new
        self.b_approx = b_new

    def clear_rec(self):
        self.a_rec = np.array([])
        self.b_rec = np.array([])

    def generate_paths(self, k, n, method = 'sgd'):
        output_a = np.zeros([k, n])
        output_b = np.zeros([k, n])
        for i in range(k):
            print(i)
            self.input_params(0.8, 0.9)
            self.opt(n, method)
            output_a[i, :] = self.a_rec
            output_b[i, :] = self.b_rec
            self.clear_rec()
        return output_a, output_b
                      
def true_func(x, a, b):
    return ((1.0/0.05**0.5)*np.exp(-(x - a)**2/0.05) +
            (1.0/0.01**0.5)*np.exp(-(x - b)**2/0.01))
 
def err(xs, a_approx, b_approx):
    
    tot = (true_func(xs, a_approx, b_approx) - true_func(xs, 0.25, 0.75))**2
    return np.mean(tot)

def err_surface(x_data, a_guess, b_guess):

    l_a = len(a_guess)
    l_b = len(b_guess)
    errors = np.zeros([l_a, l_b])
    for i in range(l_a):
        for j in range(l_b):
            errors[l_a - i - 1, j] = err(x_data, a_guess[i], b_guess[j])
    return errors

def plot_paths(method = 'sgd', runs = 5, run_length = 50):

    x_data = np.arange(0, 1, 0.01)
    a_guess = np.arange(0.0, 1.0, 0.01)
    b_guess = np.arange(0.5, 1.5, 0.01)
    errors = err_surface(x_data, a_guess, b_guess)
    Example = Optimize(0.25, 0.75, [0.8, 0.9], x_data, stochastic = True)
    a_s, b_s = Example.generate_paths(runs, run_length, method)

    plt.figure()
    plt.imshow(errors, cmap = 'Blues', interpolation='bilinear',
               extent=[0.0,1.0,0.5,1.2], vmin=0, vmax=40)
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
    
    a_guess = np.arange(0.0, 1.0, 0.01)
    b_guess = np.arange(0.5, 1.5, 0.01)

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

x_data = np.arange(0, 1, 0.01)

test2 = plot_stochastic_surface(x_data, 2)
test = plot_paths()
    
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(a_guess, b_guess, errors, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#surf = ax.plot_wireframe(a_guess, b_guess, errors)

        







