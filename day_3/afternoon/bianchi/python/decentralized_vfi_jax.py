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
    arrays = tuple(map(np.array, (m.b_grid, m.y_nodes, m.P)))
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
    H = np.reshape(b_grid, (b_size, y_size)) # Put b_grid in all cols
    return H


def T(model, v, H):
    """
    Bellman operator.

    We set up a new vector 

        W = W(b, B, y_t, y_n, bp) 

    that gives the value of the RHS of the Bellman equation at each 
    point (b, B, y_t, y_n, bp).  Then we set up a array 

        M = M(b, B, y_t, y_n, bp) 

    where 

        M(b, B, y_t, y_n, bp) = 1 if bp is feasible, else -inf.

    Then we take the max / argmax of V = W * M over the last axis.

    """
    parameters, sizes, arrays = model
    σ, η, β, ω, κ, R = parameters
    b_size, y_size = sizes
    b_grid, y_nodes = arrays

    b = np.reshape(b_grid, (b_size, 1, 1, 1))
    y = np.reshape(, (b_size, 1, 1, 1))
    
    for i_y in range(y_size):
        y_t, y_n = y_nodes[i_y]
        # Loop over aggregate state
        for i_B, B in enumerate(b_grid):
            Bp = H[i_B, i_y]
            i_Bp = np.searchsorted(b_grid, Bp)  # index corresponding to Bp
            # compute price of nontradables using aggregates
            C = R * B + y_t - Bp
            P = ((1 - ω) / ω) * (C / y_n)**(η + 1)
            # Loop over the agent's endogenous state
            for i_b, b in enumerate(b_grid):
                max_val = -np.inf
                # Search over bp choices
                for i_bp, bp in enumerate(b_grid):
                    # Impose feasibility
                    if - κ * (P * y_n + y_t) <= bp <= R * b + y_t:
                        c = R * b + y_t - bp
                        current_utility = w(model, c, y_n) 
                        continuation_val = np.sum(v[i_bp, i_Bp, :] * Q[i_y, :])
                        current_val = current_utility + β * continuation_val
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

