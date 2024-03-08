"""
Bianchi Overborrowing Model.

Python implementation of "Overborrowing and Systemic Externalities" (AER 2011)
by Javier Bianchi

In what follows

* y = (y_t, y_n) is the exogenous state process

Individual states and actions are

* c = consumption of tradables
* b = household savings (bond holdings)
* bp = b prime, household savings decision 

Aggregate quantities and prices are

* P = price of nontradables
* B = aggregate savings (bond holdings)
* C = aggregate consumption 

Vector / function versions include

* bp_vec represents bp(b, B, y) = household assets next period, etc.
* H = current guess of update rule as an array of the form H(B, y)

"""

import numpy as np
import quantecon as qe
from numba import jit, prange
from scipy.io import loadmat
from collections import namedtuple
import matplotlib.pyplot as plt


def d_infty(x, y):
    return np.max(np.abs(x - y))


Model = namedtuple('Model', 
   ('σ',            # utility parameter
    'η',            # elasticity
    'β',            # discount factor 
    'ω',            # share for tradables
    'κ',            # constraint parameter
    'R',            # gross interest rate
    'b_grid',       # bond grid
    'y_nodes',      # income nodes (each a point in R^2)
    'b_size',       # bond grid size
    'y_size',       # number of income nodes
    'Q'))           # Markov matrix


def create_overborrowing_model(
        σ=2,                 # CRRA utility parameter
        η=(1/0.83)-1,        # elasticity = 0.83, η = 0.2048
        β=0.91,              # discount factor
        ω=0.31,              # share for tradables
        κ=0.3235,            # constraint parameter
        r=0.04,              # interest rate
        b_grid_size=100,     # bond grid size, increase to 800
        b_grid_min=-1.02,    # bond grid min
        b_grid_max=-0.2      # bond grid max (originally -0.6 to match fig)
    ):    
    """
    Creates an instance of the overborrowing model using default parameter
    values from Bianchi AER 2011 with κ_n = κ_t = κ.

    The Markov matrix has the interpretation

        Q[i_y, i_yp] = one step prob of y_nodes[i_y] -> y_nodes[i_yp]

    Individual income states are extracted via (y_t, y_n) = y_nodes[i_y].

    """
    # read in Markov transitions and y grids from Bianchi's Matlab file
    data = loadmat('proc_shock.mat')
    y_t_nodes, y_n_nodes, Q = data['yT'], data['yN'], data['Prob']
    # set y[i_y] = (y_t_nodes[i_y], y_n_nodes[i_y])
    y_nodes = np.hstack((y_t_nodes, y_n_nodes))  
    # shift Q to row major, so that 
    Q = np.ascontiguousarray(Q)   
    # set up grid for bond holdings
    b_grid = np.linspace(b_grid_min, b_grid_max, b_grid_size)
    # gross interest rate
    R = 1 + r

    return Model(σ=σ, η=η, β=β, ω=ω, κ=κ, R=R, 
                 b_grid=b_grid, y_nodes=y_nodes,
                 b_size=b_grid_size, y_size=len(Q),
                 Q=Q)

@jit
def w(model, c, y_n):
    """ 
    Current utility when c_t = c and c_n = y_n.

        a = [ω c^(- η) + (1 - ω) y_n^(- η)]^(-1/η)
        w(c, y) := a^(1 - σ) / (1 - σ)

    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model
    a = (ω * c**(-η) + (1 - ω) * y_n**(-η))**(-1/η)
    return a**(1 - σ) / (1 - σ)


@jit
def generate_initial_H(model, at_constraint=False):
    """
    Compute an initial guess for H.

    Use the constraint as a the savings rule.

    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model

    H = np.empty((b_size, y_size))
    for i_B, B in enumerate(b_grid):
        for i_y, y in enumerate(y_nodes):
            y_t, y_n = y
            if at_constraint:
                c = B * R + y_t - B                  
                P = ((1 - ω) / ω) * c**(1 + η)
                H[i_B, i_y] = - κ * (P * y_n + y_t)
            else:
                H[i_B, i_y] = b_grid[i_B]  # hold steady rule
    return H


@jit(parallel=True)
def T(model, v, H):
    """
    Bellman operator.
    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model
    # Storage for new value function
    v_new = np.empty_like(v)
    # Storage for greedy policy, with the policy recorded by indices
    bp_v_greedy = np.empty_like(v)
    
    for i_y in prange(y_size):
        y_t, y_n = y_nodes[i_y]
        for i_B, B in enumerate(b_grid):
            # Updated indices and values for aggregates
            Bp = H[i_B, i_y]
            i_Bp = np.searchsorted(b_grid, Bp)
            # compute price of nontradables using aggregates
            C = R * B + y_t - Bp
            P = ((1 - ω) / ω) * (C / y_n)**(η + 1)
            # Loop
            for i_b, b in enumerate(b_grid):
                max_val = -np.inf
                # Search over bp choices
                for i_bp, bp in enumerate(b_grid):
                    # Impose feasibility
                    if - κ * (P * y_n + y_t) <= bp <= R * b + y_t:
                        c = R * b + y_t - bp
                        current_utility = w(model, c, y_n) 
                        next = β * np.sum(v[i_bp, i_Bp, :] * Q[i_y, :])
                        current_val = current_utility + next
                        if current_val > max_val:
                            max_val = current_val
                            bp_maximizer = bp
                v_new[i_b, i_B, i_y] = max_val
                bp_v_greedy[i_b, i_B, i_y] = bp_maximizer

    return v_new, bp_v_greedy


def vfi(model, H, v_init=None, max_iter=10_000, tol=1e-5, verbose=False):
    """
    Solve for the value function and update rule given H.

    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model
    error = tol + 1
    i = 0
    if v_init is None:
        v_init = np.ones((b_size, b_size, y_size))
    v = v_init

    while error > tol and i < max_iter:
        v_new, bp_policy = T(model, v, H)
        error = d_infty(v_new, v)
        v = v_new
        i += 1

    if verbose:
        print(f"VFI terminated after {i} iterations.")

    return v_new, bp_policy

def update_H(model, H, α):
    """
    Update guess of the equilibrium update rule for bonds

    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model
    H_new = np.empty_like(H)
    v_new, bp_policy = vfi(model, H, verbose=True)
    for i_B in range(b_size):
        for i_y in range(y_size):
            H_new[i_B, i_y] = α * bp_policy[i_B, i_B, i_y] + \
                             (1 - α) * H[i_B, i_y]
    return H_new


def solve_for_equilibrium(model, α=0.1, tol=0.004, max_iter=500):
    """
    Compute equilibrium law of motion.

    """
    σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model
    H = generate_initial_H(model)
    error = tol + 1
    i = 0
    while error > tol and i < max_iter:
        H_new = update_H(model, H, α)
        error = d_infty(H, H_new)
        print(f"Updated H at iteration {i} with error {error}.")
        H = H_new
        i += 1
    if i == max_iter:
        print("Warning: Equilibrium search iteration hit upper bound.")
    return H

