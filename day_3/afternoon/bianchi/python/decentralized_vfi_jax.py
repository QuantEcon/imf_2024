"""
Bianchi Overborrowing Model. See the Numba version for details.

"""

import jax
import jax.numpy as jnp
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt


def d_infty(x, y):
    return np.max(np.abs(x - y))


def convert_overborrowing_model_to_jax(numpy_model):    
    """
    Create a JAX-centric version of the overborrowing model.  Use JAX device
    arrays instead of NumPy arrays and separate data so that some components can
    be used as static arguments.

    Uses default parameters from the NumPy version.
    """
    m = numpy_model
    parameters = m.σ, m.η, m.β, m.ω, m.κ, m.R
    sizes = m.b_size, m.y_size
    arrays = tuple(map(np.array, (m.b_grid, m.y_t_nodes, m.y_n_nodes, m.P)))
    return parameters, sizes, arrays
    

def w(model, c, y_n):
    """ 
    Current utility when c_t = c and c_n = y_n.

        a = [ω c^(- η) + (1 - ω) y_n^(- η)]^(-1/η)

        w(c, y_n) := a^(1 - σ) / (1 - σ)

    """
    parameters, sizes, arrays = model
    σ, η, β, ω, κ, R = parameters
    a = (ω * c**(-η) + (1 - ω) * y_n**(-η))**(-1/η)
    return a**(1 - σ) / (1 - σ)


def generate_initial_H(model):
    """
    Compute an initial guess for H. Use a hold-steady rule.

    """
    parameters, sizes, arrays = model
    b_size, y_size = sizes
    H = np.reshape(b_grid, (b_size, y_size, y_size)) # b' = b
    return H


def T(model, v, H):
    """
    The Bellman operator.

    """
    parameters, sizes, arrays = model
    σ, η, β, ω, κ, R = parameters
    b_size, y_size = sizes
    b_grid, y_nodes = arrays

    # Expand dimension of arrays
    b   = np.reshape(b_grid,    (b_size, 1, 1, 1, 1))
    B   = np.reshape(b_grid,    (1, b_size, 1, 1, 1))
    y_t = np.reshape(y_t_nodes, (1, 1, y_size, 1, 1))
    y_n = np.reshape(y_n_nodes, (1, 1, 1, y_size, 1))
    bp  = np.reshape(b_grid,    (1, 1, 1, 1, b_size))

    # Provide some index arrays of the same shape
    b_idx   = np.reshape(range(b_size),    (b_size, 1, 1, 1, 1))

    # Construct Bp and its indices associated with H
    Bp     = np.reshape(H, (1, b_size, y_size, y_size, 1))
    Bp_idx = np.searchsorted(b_grid, Bp) 

    # compute price of nontradables using aggregates
    C = R * B + y_t - Bp
    P = ((1 - ω) / ω) * (C / y_n)**(η + 1)

    c = R * b + y_t - bp
    u = w(model, c, y_n)

    constraint_holds = - κ * (P * y_n + y_t) <= bp <= R * b + y_t

    v = np.resize(v, ?)
    EV = np.sum(v * Q, axis=?)

    W = np.where(constraint_holds, u + β * EV, -np.inf)
    v_new       = np.max(W, axis=?)
    bp_v_greedy = np.argmax(W, axis=?)

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


def solve_for_equilibrium(model, α=0.05, tol=0.004, max_iter=500):
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

