"""
Bianchi Overborrowing Model. See the Numba version for details.

"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from decentralized_vfi_numba import create_overborrowing_model, solve_for_equilibrium

#jax.config.update("jax_enable_x64", True)

@jax.jit
def d_infty(x, y):
    return jnp.max(jnp.abs(x - y))


def convert_overborrowing_model_to_jax(numpy_model):    
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
def T_generator(v, H, parameters, arrays, i_b, i_B, i_y_t, i_y_n, i_bp):
    """
    Given current state (b, B, y_t, y_n) with indices (i_b, i_B, i_y_t, i_y_n),
    compute the unmaximized right hand side (RHS) of the Bellman equation as a
    function of the next period choice bp = b'.  
    """
    σ, η, β, ω, κ, r = parameters
    b_grid, y_t_nodes, y_n_nodes, Q = arrays
    Bp = H[i_B, i_y_t, i_y_n]
    i_Bp = jnp.searchsorted(b_grid, Bp)
    y_t = y_t_nodes[i_y_t]
    y_n = y_n_nodes[i_y_n]
    B, b, bp = b_grid[i_B], b_grid[i_b], b_grid[i_bp]
    # compute price of nontradables using aggregates
    C = (1 + r) * B + y_t - Bp
    P = ((1 - ω) / ω) * (C / y_n)**(η + 1)
    c = (1 + r) * b + y_t - bp
    current_utility = w(parameters, c, y_n)
    EV = jnp.sum(v[i_bp, i_Bp, :, :] * Q[i_y_t, i_y_n, :, :])
    current_val = current_utility + β * EV
    _t = - κ * (P * y_n + y_t) <= bp
    _t1 = bp <= (1 + r) * b + y_t
    return jnp.where(jnp.logical_and(_t, _t1), current_val, -jnp.inf)


# Vectorize over the control bp and all the current states
T_vec_1 = jax.vmap(T_generator,
    in_axes=(None, None, None, None, None, None, None, None, 0))
T_vec_2 = jax.vmap(T_vec_1, 
    in_axes=(None, None, None, None, None, None, None, 0, None))
T_vec_3 = jax.vmap(T_vec_2, 
    in_axes=(None, None, None, None, None, None, 0, None, None))
T_vec_4 = jax.vmap(T_vec_3, 
    in_axes=(None, None, None, None, None, 0, None, None, None))
T_vectorized = jax.vmap(T_vec_4, 
    in_axes=(None, None, None, None, 0, None, None, None, None))


def T(parameters, sizes, arrays, v, H):
    b_size, y_size = sizes
    b_grid, y_t_nodes, y_n_nodes, Q = arrays
    b_indices, y_indices = jnp.arange(b_size), jnp.arange(y_size)
    val = T_vectorized(v, H, parameters, arrays,
                     b_indices, b_indices, y_indices, y_indices, b_indices)
    return jnp.max(val, axis=-1), b_grid[jnp.argmax(val, axis=-1)]

T = jax.jit(T, static_argnums=(1,))


def vfi(parameters, sizes, arrays, H, max_iter=10_000, tol=1e-5):
    """
    Solve for the value function and update rule given H.

    """
    b_size, y_size = sizes
    v = jnp.ones((b_size, b_size, y_size, y_size))

    def cond_fun(vals):
        error, i, v, bp = vals
        return (error > tol) & (i < max_iter)
    
    def body_fun(vals):
        _, i, v, bp = vals
        v_new, bp_policy = T(parameters, sizes, arrays, v, H)
        error = d_infty(v_new, v)
        return error, i+1, v_new, bp_policy

    error, i, v_new, bp_policy = jax.lax.while_loop(cond_fun, body_fun,
                                                    (tol+1, 0, v, v))

    return v_new, bp_policy, i

vfi = jax.jit(vfi, static_argnums=(1,))


def update_H(parameters, sizes, arrays, H, α):
    """
    Update guess of the equilibrium update rule for bonds

    """
    b_size, y_size = sizes
    _, bp_policy, vfi_num_iter = vfi(parameters, sizes, arrays, H)
    b_indices = jnp.arange(b_size)
    new_H = α * bp_policy[b_indices, b_indices, :, :] + (1 - α) * H
    return new_H, vfi_num_iter

update_H = jax.jit(update_H, static_argnums=(1,))


def solve_for_equilibrium_jax(parameters, sizes, arrays,
                          α=0.05, tol=0.004, max_iter=500):
    """
    Compute equilibrium law of motion.

    """
    H = generate_initial_H(parameters, sizes, arrays)
    error = tol + 1
    i = 0
    while error > tol and i < max_iter:
        H_new, vfi_num_iter = update_H(parameters, sizes, arrays, H, α)
        print(f"VFI terminated after {vfi_num_iter} iterations.")
        error = d_infty(H, H_new)
        print(f"Updated H at iteration {i} with error {error}.")
        H = H_new
        i += 1
    if i == max_iter:
        print("Warning: Equilibrium search iteration hit upper bound.")
    return H

numpy_model = create_overborrowing_model()
parameters, sizes, arrays = convert_overborrowing_model_to_jax(numpy_model)
jax_in_time = time.time()
H_jax = solve_for_equilibrium_jax(parameters, sizes, arrays)
jax_out_time = time.time()

np_in_time = time.time()
#H_np = solve_for_equilibrium(numpy_model)
np_out_time = time.time()

print("JAX time:", jax_out_time-jax_in_time)
print("numpy time:", np_out_time-np_in_time)

# Uncomment for plotting
b_size, y_size = sizes
b_grid, y_t_nodes, y_n_nodes, Q = arrays

fig, ax = plt.subplots()
for i_y in range(y_size): 
    ax.plot(b_grid, H_jax[:, i_y])
ax.plot(b_grid, b_grid, color='black', ls='--')
plt.show()

