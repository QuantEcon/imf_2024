"""
Bianchi Overborrowing Model.

Python implementation of "Overborrowing and Systemic Externalities" (AER 2011)
by Javier Bianchi

In what follows

* y = (y_t, y_n)
* c = consumption of tradables
* p = price of nontradables
* bp = b prime, bonds next period

The vector / function version is

* c_vec represents c(b, y) = consumption function for tradables
* p_vec represents (b, y) = price function for nontradables
* bp_vec represents (b, y) = b prime, bonds next period


Interpolation of f(b, y) is done over b -> f(b, y) for each y.

"""

import scipy as sp
import numpy as np
import pandas as pd
import quantecon as qe
from numba import njit
from scipy.io import loadmat
from scipy.optimize import root, newton
from collections import namedtuple
import matplotlib.pyplot as plt


def d_infty(x, y):
    return np.max(np.abs(x - y))


Model = namedtuple('Model', 
   ('σ', 'η', 'β', 'ω', 'κ', 'R', 'b_grid', 'y_nodes', 'P'))


def create_overborrowing_model(
        σ=2,          # CRRA utility parameter
        η=(1/0.83)-1, # elasticity (elasticity = 0.83)
        β=0.91,       # discount factor
        ω=0.31,       # share for tradables
        κ=0.3235,     # constraint parameter
        r=0.04,       # interest rate
        b_grid_size=100,     # bond grid size, increase to 800
        b_grid_min=-1.02,    # bond grid min
        b_grid_max=-0.6      # bond grid max
    ):    
    """
    Creates an instance of the overborrowing model using default parameter
    values from Bianchi AER 2011.  He sets κ_n = κ_t = κ.

    The bivariate nodes for y and the corresponding Markov matrix P have the
    form

        P[i_y, i_yp] = one step prob of y_nodes[i_y] -> y_nodes[i_yp]

    Individual income states are extracted via (y_t, y_n) = y_nodes[i_y].

    """
    # read in Markov transitions and y grids from Bianchi's Matlab file
    data = loadmat('proc_shock.mat')
    y_t_nodes, y_n_nodes, P = data['yT'], data['yN'], data['Prob']
    # set y[i_y] = (y_t_nodes[i_y], y_n_nodes[i_y])
    y_nodes = np.hstack((y_t_nodes, y_n_nodes))  
    # shift P to row major, so that 
    P = np.ascontiguousarray(P)   
    # set up grid for bond holdings
    b_grid = np.linspace(b_grid_min, b_grid_max, b_grid_size)
    # gross interest rate
    R = 1 + r

    return Model(σ=σ, η=η, β=β, ω=ω, κ=κ, R=R, 
                 b_grid=b_grid, y_nodes=y_nodes, P=P)


def linear_function_factory(model, values):
    """
    Return the function f such that

        f(b, i) = linear interpolation over (b_grid, values[:, i])
        
    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, P = model

    def f(b, i):
        return np.interp(b, b_grid, values[:, i])
    return f


def p(model, i_y, c):
    """
    Price of nontradables given (y_t, y_n) = y[i_y] and tradables consumption c.
    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, P = model
    y_t, y_n = y_nodes[i_y]
    return ((1 - ω) / ω) * (c / y_n)**(η + 1)


def c_given_bond_price(model, b, i_y, bp):
    """
    Consumption of tradables c given b, (y_t, y_n) = y[i_y] and bp. 
    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, P = model
    y_t, y_n = y_nodes[i_y]
    return b * R + y_t - bp


def bp_bind(model, i_y, p):
    """
    Constrained bond price given (y_t, y_n) = y[i_y] and p.
    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, P = model
    y_t, y_n = y_nodes[i_y]
    return - κ * (p * y_n + y_t)


def mu(model, c, i_y):
    """
    Compute marginal utility 

        m(c, y) := [ω c^(- η) + (1 - ω) y_n^(- η)]^((σ - 1)/η - 1) ω c^(- η - 1)
            
    given (y_t, y_n) = y[i_y] and tradables consumption c.

    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, P = model
    y_t, y_n = y_nodes[i]
    a = ω * c**(-η) + (1 - ω) * y_n**(-η)
    return a**((σ-1)/η - 1) * ω * c**(-η-1)


def exp_mu(model, b, i_y, c_vec):
    """
    Compute expected marginal utility 

        Σ_{i_yp | i_y} mu(c(bp, i_yp), i_yp) P[i_y, i_py]
            
    where c := tradables consumption.

    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, P = model
    y_t, y_n = y_nodes[i]
    total = 0.0
    for j in len(y_nodes):
        total += mu(c_func(bp, j), j) * P[i, j]
    return total



def update_all(model, c_vec, bp_vec):
    """
    Update the functions c, p and bp.  Read them in as vectors
    of the form

        v[i_b, i_y] = value of the function at (b_grid[i_b], y_nodes[i_y])

    Return them in the same format.

    """
    # Unpack
    σ, η, β, ω, κ, R, b_grid, y_nodes, P = model
    y_size, b_size = len(y_nodes), len(b_size)
    # Set up storage and functions
    vecs = p_vec, c_vec, bp_vec
    new_p, new_c, new_bp = [np.empty_like(v) for v in vecs]
    p_func, c_func, bp_func = \
            [linear_function_factory(model, v) for v in vecs]

    for i_b in range(b_size):
        for i_y in range(y_size):
            b = b_grid[i_b]
            p = p_vec[i_b, i_y] 
            b_bind = bp_bind(model, y_idx, p)
            c_bind = c_given_bond_price(model, b, y_idx, bp)
            u_t = mu(model, b, y_idx, c_bind)
            u_t_prime = β * R * mu(model, b_bind, y_idx, c)






if False:

    def corr(x, y):
        m_x, m_y = x.mean(), y.mean()
        s_xy = np.sqrt(np.sum((x - m_x)**2) * np.sum((y - m_y)**2))
        return np.sum((x - m_x) * (y - m_y)) / (s_xy)

    def generate_income_mc(grid_size_1=4, grid_size_2=4, test=False):
        rng = np.random.default_rng(12345)
        A = [[0.901,   0.495],
            [ -0.453,  0.225]]
        Ω = [[0.00219, 0.00162],
            [ 0.00162, 0.00167]]
        C = sp.linalg.sqrtm(Ω)
        grid_sizes = grid_size_1, grid_size_2
        mc = qe.markov.discrete_var(A, C, grid_sizes, 
                                    sim_length=1_000_000,
                                    std_devs=np.sqrt(3),
                                    random_state=rng)
        if test:
            y_series = mc.simulate(1_000_000, random_state=rng)
            y_t_series = np.exp(y_series[:, 0])
            y_n_series = np.exp(y_series[:, 1])
            print(f"Std dev of y_t is {y_t_series.std()}")
            print(f"Std dev of y_n is {y_n_series.std()}")
            print(f"corr(y_t, y_n) is {corr(y_t_series, y_n_series)}")
            print(f"auto_corr(y_t) is {corr(y_t_series[:-1], y_t_series[1:])}")
            print(f"auto_corr(y_n) is {corr(y_n_series[:-1], y_n_series[1:])}")

        y_states = np.exp(mc.state_values)
        #return(y_states, mc.P)


    def test_loadmat():
        # read in Markov transitions and y grids from Bianchi's Matlab file
        data = loadmat('proc_shock.mat')
        y_t_nodes, y_n_nodes, P = data['yT'], data['yN'], data['Prob']
        y = np.hstack((y_t_nodes, y_n_nodes))  # y[i] = (y_t_nodes[i], y_n_nodes[i])
        P = np.ascontiguousarray(P)          # P[i, j] = Prob y[i] -> y[j]           
        mc = qe.MarkovChain(P, y)
        X = mc.simulate(ts_length=1_000_000, init=y[0])
        y_t_series = X[:, 0]
        y_n_series = X[:, 1]
        print(f"Std dev of y_t is {y_t_series.std()}")
        print(f"Std dev of y_n is {y_n_series.std()}")
        print(f"corr(y_t, y_n) is {corr(y_t_series, y_n_series)}")
        print(f"auto_corr(y_t) is {corr(y_t_series[:-1], y_t_series[1:])}")
        print(f"auto_corr(y_n) is {corr(y_n_series[:-1], y_n_series[1:])}")

