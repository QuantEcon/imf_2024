"""
Bianchi Overborrowing Model. See the Numba version for details.

"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from decentralized_vfi_numba import create_overborrowing_model


@jax.jit
def d_infty(x, y):
    return jnp.max(jnp.abs(x - y))


def convert_overborrowing_model_to_jax(numpy_model=create_overborrowing_model()):    
    """
    Create a JAX-centric version of the overborrowing model.  Use JAX device
    arrays instead of NumPy arrays and separate data so that some components can
    be used as static arguments.

    Uses default parameters from the NumPy version.
    """
    m = numpy_model
    parameters = m.σ, m.η, m.β, m.ω, m.κ, m.r
    sizes = m.b_size, m.y_size
    arrays = tuple(map(jnp.array, (m.b_grid, m.y_t_nodes, m.y_n_nodes, m.Q)))
    return parameters, sizes, arrays


@jax.jit
def w(parameters, c, y_n):
    """ 
    Current utility when c_t = c and c_n = y_n.

        a = [ω c^(- η) + (1 - ω) y_n^(- η)]^(-1/η)

        w(c, y_n) := a^(1 - σ) / (1 - σ)

    """
    σ, η, β, ω, κ, r = parameters
    a = (ω * c**(-η) + (1 - ω) * y_n**(-η))**(-1/η)
    return a**(1 - σ) / (1 - σ)


@jax.jit
def _H_at_constraint(parameters, B, y_t, y_n):
    σ, η, β, ω, κ, r = parameters
    c = B * (1 + r) + y_t - B                  
    P = ((1 - ω) / ω) * c**(1 + η)
    return - κ * (P * y_n + y_t)

_H_at_constraint_n = jax.vmap(_H_at_constraint,
                              in_axes=(None, None, None, 0))
_H_at_constraint_n_t = jax.vmap(_H_at_constraint_n,
                              in_axes=(None, None, 0, None))
_H_at_constraint_n_t_B = jax.vmap(_H_at_constraint_n_t,
                              in_axes=(None, 0, None, None))



def _H_no_constraint(B, sizes):
    b_size, y_size = sizes
    return jnp.full((y_size, y_size), B)

_H_no_constraint = jax.jit(_H_no_constraint, static_argnums=(1,))
_H_no_constraint_B = jax.vmap(_H_no_constraint,
                              in_axes=(0, None))



def generate_initial_H(parameters, sizes, arrays, at_constraint=False):
    """
    Compute an initial guess for H. Use a hold-steady rule.

    """
    b_grid, y_t_nodes, y_n_nodes, Q = arrays
    if at_constraint:
        return _H_at_constraint_n_t_B(parameters, b_grid, y_t_nodes, y_n_nodes)
    return _H_no_constraint_B(b_grid, sizes)

@jax.jit
def T_gen(v, H, parameters, arrays, i_b, i_B, i_y_t, i_y_n):
    σ, η, β, ω, κ, r = parameters
    b_grid, y_t_nodes, y_n_nodes, Q = arrays
    Bp = H[i_B, i_y_t, i_y_n]
    i_Bp = jnp.searchsorted(b_grid, Bp)
    y_t = y_t_nodes[i_y_t]
    y_n = y_n_nodes[i_y_n]
    b = b_grid[i_b]
    # compute price of nontradables using aggregates
    C = (1 + r) * b_grid[i_B] + y_t - Bp
    P = ((1 - ω) / ω) * (C / y_n)**(η + 1)
    c = (1 + r) * b + y_t - b_grid
    current_utility = w(parameters, c, y_n)
    EV = jnp.sum(v[:, i_Bp, :, :] * Q[i_y_t, i_y_n, :, :], axis=(1, 2))
    current_val = current_utility + β * EV
    _t = - κ * (P * y_n + y_t) <= b_grid
    _t1 = b_grid <= (1 + r) * b + y_t
    return jnp.where(_t & _t1, current_val, -jnp.inf)

T_gen_n = jax.vmap(T_gen, in_axes=(None, None, None, None, None, None, None, 0))
T_gen_n_t = jax.vmap(T_gen_n, in_axes=(None, None, None, None, None, None, 0, None))
T_gen_y_n_B = jax.vmap(T_gen_n_t, in_axes=(None, None, None, None, None, 0, None, None))
T_gen_y_n_B_b = jax.vmap(T_gen_y_n_B, in_axes=(None, None, None, None, 0, None, None, None))


def T(parameters, sizes, arrays, v, H):
    b_size, y_size = sizes
    val = T_gen_y_n_B_b(v, H, parameters, arrays,
                         jnp.arange(b_size), jnp.arange(b_size),
                         jnp.arange(y_size), jnp.arange(y_size))
    return jnp.max(val, axis=-1), jnp.argmax(val, axis=-1)

T = jax.jit(T, static_argnums=(1,))


def vfi(parameters, sizes, arrays, H, max_iter=10_000, tol=1e-5):
    """
    Solve for the value function and update rule given H.

    """
    b_size, y_size = sizes
    v = jnp.ones((b_size, b_size, y_size, y_size))
    bp_policy_init =jnp.zeros((b_size, b_size, y_size, y_size), dtype=jnp.int32)

    def cond_fun(vals):
        error, i, v, bp = vals
        return (error > tol) & (i < max_iter)
    
    def body_fun(vals):
        _, i, v, bp = vals
        v_new, bp_policy = T(parameters, sizes, arrays, v, H)
        error = d_infty(v_new, v)
        return error, i+1, v_new, bp_policy

    error, i, v_new, bp_policy = jax.lax.while_loop(cond_fun, body_fun,
                                                    (tol+1, 0, v, bp_policy_init))

    return v_new, bp_policy

vfi = jax.jit(vfi, static_argnums=(1,))

def update_H(parameters, sizes, arrays, H, α):
    """
    Update guess of the equilibrium update rule for bonds

    """
    b_size, y_size = sizes
    _, bp_policy = vfi(parameters, sizes, arrays, H)
    b_range = jnp.arange(b_size)
    return α * bp_policy[b_range, b_range,:, :] + (1 - α) * H

update_H = jax.jit(update_H, static_argnums=(1,))


def solve_for_equilibrium(α=0.05, tol=0.004, max_iter=500):
    """
    Compute equilibrium law of motion.

    """
    parameters, sizes, arrays = convert_overborrowing_model_to_jax()
    H = generate_initial_H(parameters, sizes, arrays)
    error = tol + 1
    i = 0
    while error > tol and i < max_iter:
        H_new = update_H(parameters, sizes, arrays, H, α)
        error = d_infty(H, H_new)
        print(f"Updated H at iteration {i} with error {error}.")
        H = H_new
        i += 1
    if i == max_iter:
        print("Warning: Equilibrium search iteration hit upper bound.")
    return H

solve_for_equilibrium()