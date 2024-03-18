"""
Bianchi Overborrowing Model.

See the JAX version for details.

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
   ('σ',            # Utility parameter
    'η',            # Elasticity
    'β',            # Discount factor 
    'ω',            # Share for tradables
    'κ',            # Constraint parameter
    'r',            # Interest rate
    'b_grid',       # Bond grid
    'y_t_nodes',    # Income nodes for y_t
    'y_n_nodes',    # Income nodes for y_n
    'b_size',       # Bond grid size
    'y_size',       # Number of income nodes along each dimension
    'Q'))           # Markov matrix


def create_overborrowing_model(
        σ=2,                 # CRRA utility parameter
        η=(1/0.83)-1,        # Elasticity = 0.83, η = 0.2048
        β=0.91,              # Discount factor
        ω=0.31,              # Share for tradables
        κ=0.3235,            # Constraint parameter
        r=0.04,              # Interest rate
        b_size=250,          # Bond grid size
        b_grid_min=-1.02,    # Bond grid min
        b_grid_max=-0.2      # Bond grid max (originally -0.6 to match fig)
    ):    
    """
    Creates an instance of the overborrowing model using default parameter
    values from Bianchi AER 2011 with κ_n = κ_t = κ.

    The Markov kernel Q has the interpretation

        Q[i, j, ip, jp] = one step prob of moving from 
                            (y_t[i], y_n[j]) to (y_t[ip], y_n[jp])

    """
    # Read in data using parameters estimated in Yamada (2023)
    y_t_nodes, y_n_nodes, Q = discretize_income_var(single_index=False)
    # Set up grid for bond holdings
    b_grid = np.linspace(b_grid_min, b_grid_max, b_size)

    return Model(σ=σ, η=η, β=β, ω=ω, κ=κ, r=r, 
                 b_grid=b_grid, 
                 y_t_nodes=y_t_nodes, y_n_nodes=y_n_nodes,
                 b_size=b_size, y_size=len(Q), Q=Q)

@jit
def w(model, c, y_n):
    """ 
    Current utility when c_t = c and c_n = y_n.

        a = [ω c^(- η) + (1 - ω) y_n^(- η)]^(-1/η)
        w(c, y) := a^(1 - σ) / (1 - σ)

    """
    σ, η, β, ω, κ, r, b_grid, y_t_nodes, y_n_nodes, b_size, y_size, Q = model
    a = (ω * c**(-η) + (1 - ω) * y_n**(-η))**(-1/η)
    return a**(1 - σ) / (1 - σ)


@jit
def generate_initial_H(model, at_constraint=False):
    """
    Compute an initial guess for H.

    Use the constraint as a the savings rule.

    """
    σ, η, β, ω, κ, r, b_grid, y_t_nodes, y_n_nodes, b_size, y_size, Q = model

    H = np.empty((b_size, y_size, y_size))
    for i_B, B in enumerate(b_grid):
        for i_y_t, y_t in enumerate(y_t_nodes):
            for i_y_n, y_n in enumerate(y_n_nodes):
                if at_constraint:
                    c = B * (1 + r) + y_t - B                  
                    P = ((1 - ω) / ω) * c**(1 + η)
                    H[i_B, i_y_t, i_y_n] = - κ * (P * y_n + y_t)
                else:
                    H[i_B, i_y_t, i_y_n] = b_grid[i_B]  # hold steady rule
    return H


@jit(parallel=True)
def T(model, v, H):
    """
    Bellman operator

        Tv(b, B, y_t, y_n) = max_{b'} { 
            w(c, y_n) + 
            Σ_{y_t', y_n'} v(b', H(B, y), y_t', y_n') Q(y_t, y_n, y_t', y_n')
          }

    subject to

        c = (1 + r) b + y_t - b'

    and 

        - κ (P y_n + y_t) <= b' <= (1 + r) b + y_t
    
    where prices are determined by

        C = (1 + r) * B + y_t - Bp
        P = ((1 - ω) / ω) * (C / y_n)**(η + 1)

    The value function indices are written as

        v = v[i_b, i_B, i_y_t, i_y_n]

    """
    σ, η, β, ω, κ, r, b_grid, y_t_nodes, y_n_nodes, b_size, y_size, Q = model
    # Storage for new value function
    v_new = np.empty_like(v)
    # Storage for greedy policy
    bp_v_greedy = np.empty_like(v)
    
    for i_y_t in prange(y_size):
        y_t = y_t_nodes[i_y_t]
        for i_y_n in range(y_size):
            y_n = y_n_nodes[i_y_n]
            for i_B, B in enumerate(b_grid):
                # Updated indices and values for aggregates
                Bp = H[i_B, i_y_t, i_y_n]
                i_Bp = np.searchsorted(b_grid, Bp)
                # compute price of nontradables using aggregates
                C = (1 + r) * B + y_t - Bp
                P = ((1 - ω) / ω) * (C / y_n)**(η + 1)
                # Loop
                for i_b, b in enumerate(b_grid):
                    max_val = -np.inf
                    # Search over bp choices
                    for i_bp, bp in enumerate(b_grid):
                        # Impose feasibility
                        if - κ * (P * y_n + y_t) <= bp <= (1 + r) * b + y_t:
                            c = (1 + r) * b + y_t - bp
                            current_utility = w(model, c, y_n) 
                            # Compute expected value tomorrow
                            exp = 0.0
                            for i_y_tp in range(y_size):
                                for i_y_np in range(y_size):
                                    exp += v[i_bp, i_Bp, i_y_tp, i_y_np] \
                                            * Q[i_y_t, i_y_n, i_y_tp, i_y_np]
                            current_val = current_utility + β * exp
                            if current_val > max_val:
                                max_val = current_val
                                bp_maximizer = bp
                    v_new[i_b, i_B, i_y_t, i_y_n] = max_val
                    bp_v_greedy[i_b, i_B, i_y_t, i_y_n] = bp_maximizer

    return v_new, bp_v_greedy


def vfi(model, H, v_init=None, max_iter=10_000, tol=1e-5, verbose=False):
    """
    Solve for the value function and update rule given H.

    """
    σ, η, β, ω, κ, r, b_grid, y_t_nodes, y_n_nodes, b_size, y_size, Q = model
    error = tol + 1
    i = 0
    if v_init is None:
        v_init = np.ones((b_size, b_size, y_size, y_size))
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
    σ, η, β, ω, κ, r, b_grid, y_t_nodes, y_n_nodes, b_size, y_size, Q = model
    H_new = np.empty_like(H)
    v_new, bp_policy = vfi(model, H, verbose=True)
    for i_B in range(b_size):
        for i_y_t in range(y_size):
            for i_y_n in range(y_size):
                H_new[i_B, i_y_t, i_y_n] = α * \
                        bp_policy[i_B, i_B, i_y_t, i_y_n] + \
                             (1 - α) * H[i_B, i_y_t, i_y_n]
    return H_new


def solve_for_equilibrium(model, α=0.1, tol=0.004, max_iter=500):
    """
    Compute equilibrium law of motion.

    """
    σ, η, β, ω, κ, r, b_grid, y_t_nodes, y_n_nodes, b_size, y_size, Q = model
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


#m = create_overborrowing_model()
#σ, η, β, ω, κ, r, b_grid, y_t_nodes, y_n_nodes, b_size, y_size, Q = m
#solve_for_equilibrium(m)
