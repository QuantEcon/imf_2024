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

# Auxillary functions

def single_to_multi(m, n):
    return m // n, m % n

def multi_to_single(i, j, n):
    return n * i + j

def corr(x, y):
    m_x, m_y = x.mean(), y.mean()
    s_xy = np.sqrt(np.sum((x - m_x)**2) * np.sum((y - m_y)**2))
    return np.sum((x - m_x) * (y - m_y)) / (s_xy)


# Main functions

def generate_original(A=A2, Ω=Ω2, sim_length=1_000_000):
    C = sp.linalg.sqrtm(Ω)
    y_series = np.empty((sim_length, 2))
    y_series[0, :] = np.zeros(2)
    for t in range(sim_length-1):
        y_series[t+1, :] = A @ y_series[t, :] + C @ np.random.randn(2)
    y_t_series = np.exp(y_series[:, 0])
    y_n_series = np.exp(y_series[:, 1])
    return y_t_series, y_n_series


def discretize_income_var(A=A2, Ω=Ω2, 
                          grid_size=4, 
                          single_index=True):
    C = sp.linalg.sqrtm(Ω)
    rng = np.random.default_rng(12345)
    mc = qe.markov.discrete_var(A, C, 
                                (grid_size, grid_size),
                                sim_length=1_000_000,
                                std_devs=np.sqrt(3),
                                random_state=rng)
    y_nodes, Q = np.exp(mc.state_values), mc.P
    if single_index:
        return y_nodes, Q
    else:
        y_t_nodes = y_nodes[:, 0]
        y_n_nodes = y_nodes[:, 1]
        n = grid_size
        Q_multi = np.empty((n, n, n, n))
        for i in range(n):
            for j in range(n):
                m = multi_to_single(i, j, n)
                for ip in range(n):
                    for jp in range(n):
                        mp = multi_to_single(ip, jp, n)
                        Q_multi[i, j, ip, jp] = Q[m, mp]
        y_t_nodes_multi = [y_t_nodes[n * i] for i in range(n)]
        y_n_nodes_multi = y_n_nodes[0: 4]
    return y_t_nodes_multi, y_n_nodes_multi, Q_multi


def generate_income_mc(mc, sim_length=1_000_000):
    y_series = mc.simulate(sim_length, random_state=rng)
    y_t_series = np.exp(y_series[:, 0])
    y_n_series = np.exp(y_series[:, 1])
    return y_t_series, y_n_series


def load_bianchi_matrix(simulate=True):
    # Read in Markov transitions and y grids from Bianchi's Matlab file
    data = loadmat('proc_shock.mat')
    y_t_nodes, y_n_nodes, P = data['yT'], data['yN'], data['Prob']
    y = np.hstack((y_t_nodes, y_n_nodes))  # y[i] = (y_t_nodes[i], y_n_nodes[i])
    # shift to row major
    P = np.ascontiguousarray(P)            # P[i, j] = Prob y[i] -> y[j]           
    if not simulate:
        return y, P
    else:
        mc = qe.MarkovChain(P, y)
        X = mc.simulate(ts_length=1_000_000, init=y[0])
        y_t_series = X[:, 0]
        y_n_series = X[:, 1]
        return y_t_series, y_n_series


def print_stats(y_t_series, y_n_series):
    print(f"Std dev of y_t is {y_t_series.std()}")
    print(f"Std dev of y_n is {y_n_series.std()}")
    print(f"corr(y_t, y_n) is {corr(y_t_series, y_n_series)}")
    print(f"auto_corr(y_t) is {corr(y_t_series[:-1], y_t_series[1:])}")
    print(f"auto_corr(y_n) is {corr(y_n_series[:-1], y_n_series[1:])}")
