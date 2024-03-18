"""
Functions for working with the VAR process

    y' = A y + u'   (prime indicates next period value)

where

    * y = (y_t, y_n) = (tradables, nontradables)
    * A is 2 x 2
    * u' ~ N(0, Ω)

"""

import numpy as np
from scipy.io import loadmat
import pandas as pd
import quantecon as qe
import scipy as sp
import matplotlib.pyplot as plt


# Reported in Bianchi (2011)

A1 = [[0.901,   0.495],
      [-0.453,  0.225]]
Ω1 = [[0.00219, 0.00162],
      [0.00162, 0.00167]]

# Reported in Yamada (2023)
# p. 12 of https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/514

A2 = [[0.2425,   0.3297],
      [-0.1984,  0.7576]]
Ω2 = [[0.0052, 0.002],
      [0.002,  0.0059]]


## == Main function == ##

def discretize_income_var(A=A2, Ω=Ω2, grid_size=4, seed=1234):
    """
    Discretize the VAR model, returning

        y_t_nodes, a grid of y_t values
        y_n_nodes, a grid of y_n values
        Q, a Markov operator

    Let n = grid_size. The format is that Q is n x n x n x n, with

        Q[i, j, i', j'] = one step transition prob from 
        (y_t_nodes[i], y_n_nodes[j]) to (y_t_nodes[i'], y_n_nodes[j'])

    """
    
    C = sp.linalg.sqrtm(Ω)
    n = grid_size
    rng = np.random.default_rng(seed)
    mc = qe.markov.discrete_var(A, C, (n, n),
                                sim_length=1_000_000,
                                std_devs=np.sqrt(3),
                                random_state=rng)
    y_nodes, Q = np.exp(mc.state_values), mc.P
    # The array y_nodes is currently an array listing all 2 x 1 state pairs
    # (y_t, y_n), so that y_nodes[i] is the i-th such pair, while Q[l, m] 
    # is the probability of transitioning from state l to state m in one step. 
    # We switch the representation to the one described in the docstring.
    y_t_nodes = [y_nodes[n*i, 0] for i in range(n)]  
    y_n_nodes = y_nodes[0:4, 1]                      
    Q = np.reshape(Q, (n, n, n, n))
    return y_t_nodes, y_n_nodes, Q


## == Tests == #

def generate_var_process(A=A2, Ω=Ω2, sim_length=1_000_000):
    """
    Generate the original VAR process.

    """
    C = sp.linalg.sqrtm(Ω)
    y_series = np.empty((sim_length, 2))
    y_series[0, :] = np.zeros(2)
    for t in range(sim_length-1):
        y_series[t+1, :] = A @ y_series[t, :] + C @ np.random.randn(2)
    y_t_series = np.exp(y_series[:, 0])
    y_n_series = np.exp(y_series[:, 1])
    return y_t_series, y_n_series

def corr(x, y):
    m_x, m_y = x.mean(), y.mean()
    s_xy = np.sqrt(np.sum((x - m_x)**2) * np.sum((y - m_y)**2))
    return np.sum((x - m_x) * (y - m_y)) / (s_xy)


def print_stats(y_t_series, y_n_series):
    print(f"Std dev of y_t is {y_t_series.std()}")
    print(f"Std dev of y_n is {y_n_series.std()}")
    print(f"corr(y_t, y_n) is {corr(y_t_series, y_n_series)}")
    print(f"auto_corr(y_t) is {corr(y_t_series[:-1], y_t_series[1:])}")
    print(f"auto_corr(y_n) is {corr(y_n_series[:-1], y_n_series[1:])}")
