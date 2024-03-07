import numpy as np
from scipy.io import loadmat
import pandas as pd
import quantecon as qe

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

