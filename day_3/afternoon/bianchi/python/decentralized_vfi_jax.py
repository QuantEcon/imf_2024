"""
Bianchi Overborrowing Model.

Python/JAX implementation of "Overborrowing and Systemic Externalities" 
(AER 2011) by Javier Bianchi

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

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mc_dynamics import discretize_income_var

#jax.config.update("jax_enable_x64", True)

@jax.jit
def d_infty(x, y):
    return jnp.max(jnp.abs(x - y))


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
    y_t_nodes, y_n_nodes, Q = discretize_income_var()
    # Shift to JAX arrays
    y_t_nodes, y_n_nodes, Q = tuple(map(jnp.array, 
                                        (y_t_nodes, y_n_nodes, Q)))
    # Set up grid for bond holdings
    b_grid = jnp.linspace(b_grid_min, b_grid_max, b_size)

    parameters = σ, η, β, ω, κ, r
    sizes = b_size, len(y_t_nodes)
    arrays = b_grid, y_t_nodes, y_n_nodes, Q
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
    utility = w(parameters, c, y_n)
    EV = jnp.sum(v[i_bp, i_Bp, :, :] * Q[i_y_t, i_y_n, :, :])
    credit_constraint_holds = - κ * (P * y_n + y_t) <= bp
    budget_constraint_holds = bp <= (1 + r) * b + y_t
    return jnp.where(jnp.logical_and(credit_constraint_holds, 
                                     budget_constraint_holds), 
                     utility + β * EV,
                     -jnp.inf)


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
    # Evaluate RHS of Bellman equation at all states and actions
    val = T_vectorized(v, H, parameters, arrays,
                     b_indices, b_indices, y_indices, y_indices, b_indices)
    # Maximize over bp
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
    Update guess of the aggregate update rule.

    """
    b_size, y_size = sizes
    b_indices = jnp.arange(b_size)
    # Compute household response to current guess H
    v, bp_policy, vfi_num_iter = vfi(parameters, sizes, arrays, H)
    # Update guess
    new_H = α * bp_policy[b_indices, b_indices, :, :] + (1 - α) * H
    return new_H, vfi_num_iter

update_H = jax.jit(update_H, static_argnums=(1,))


def compute_equilibrium(parameters, sizes, arrays,
                          α=0.05, tol=0.004, max_iter=500):
    """
    Compute the equilibrium law of motion.

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


## Test

model = create_overborrowing_model()
parameters, sizes, arrays = model
b_size, y_size = sizes
b_grid, y_t_nodes, y_n_nodes, Q = arrays

jax_in_time = time.time()
H_jax = compute_equilibrium(parameters, sizes, arrays)
jax_out_time = time.time()

print("JAX time:", jax_out_time - jax_in_time)

fig, ax = plt.subplots()
for i_y in range(y_size): 
    ax.plot(b_grid, H_jax[:, i_y])
ax.plot(b_grid, b_grid, color='black', ls='--')
plt.show()

