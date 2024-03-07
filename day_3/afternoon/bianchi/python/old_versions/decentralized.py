# # Bianchi Overborrowing Model.
#
# Python implementation of 
#
# * Overborrowing and Systemic Externalities (AER 2011) by Javier Bianchi

import numpy as np
from numba import njit
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import root, newton
from collections import namedtuple
import matplotlib.pyplot as plt


def d_infty(x, y):
    return np.max(np.abs(x - y))

Model = namedtuple('Model', 
   ('σ', 'η', 'β', 'ω', 'κ', 'R', 'b_grid', 'yn_grid', 'yt_grid', 'P'))

def create_overborrowing_model(σ=2,          # CRRA utility parameter
                               η=(1/0.83)-1, # elasticity (elasticity = 0.83)
                               β=0.91,       # discount factor
                               ω=0.31,       # share for tradables
                               κ=0.3235,     # constraint parameter
                               r=0.04,       # interest rate
                               n_B=100,      # bond grid size
                               b_grid_min=-1.02,    # grid min
                               b_grid_max=-0.6):    # grid max
    """
    Creates an instance of the overborrowing model using default parameter
    values from Bianchi AER 2011.

    """

    # Read in Markov transitions and y grids from Bianchi's Matlab file
    markov_data = loadmat('proc_shock.mat')
    yt_grid, yn_grid, P = markov_data['yT'], markov_data['yN'], markov_data['Prob']

    P = np.ascontiguousarray(P)   # shift P to row major
    # Set up grid for bond holdings
    b_grid = np.linspace(b_grid_min, b_grid_max, n_B)

    return Model(σ=σ, η=η, β=β, ω=ω, κ=κ, R=(1 + r), 
                 b_grid=b_grid, yn_grid=yn_grid, yt_grid=yt_grid, P=P)



def initialize_decentralized_eq_search(model):

    # Unpack
    σ, η, β, ω, κ, R, b_grid, yn_grid, yt_grid, P = model
    n_B =  len(b_grid)
    n_y =  len(yn_grid)

    # Reshape
    b  = np.reshape(b_grid,  (n_B, 1))
    yt = np.reshape(yt_grid, (1, n_y))
    yn = np.reshape(yn_grid, (1, n_y))
    bp = np.tile(b, (1, n_y))            # initial guess for bp

    c = b * R + yt - bp                  
    price = ((1 - ω) / ω) * c**(1 + η)
    b_bind  = -κ * (price * yn + yt)
    c_bind = b * R + yt - b_bind

    return c, c_bind, bp


@njit
def compute_marginal_utility(c, model):
    # Unpack and set up
    σ, η, β, ω, κ, R, b_grid, yn_grid, yt_grid, P = model
    n_y =  len(yn_grid)
    yn = np.reshape(yn_grid, (1, n_y))
    # Compute an aggregated consumption good assuming c_N = y_N
    total_c = ω * c**(-η) + (1-ω) * yn**(-η)
    # Compute marginal utility 
    mup = ω * total_c**(σ/η-1/η-1) * (c**(-η-1))  
    return mup


def compute_exp_marginal_utility(mu, bp, model):
    # Unpack and set up
    σ, η, β, ω, κ, R, b_grid, yn_grid, yt_grid, P = model
    n_B, n_y =  len(b_grid), len(yn_grid)
    # Allocate memory
    exp_mu = np.empty((n_B, n_y)) 
    mu_vals = np.empty(n_y)

    f = interp1d(b_grid, mu, axis=0, bounds_error=False,
                 fill_value='interpolate')
    # Compute expected marginal utility in today's grid
    for i in range(n_B):
        for j in range(n_y):
            exp_mu[i, j] =  β * R * f(bp[i, j]) @ P[j, :]
    return exp_mu


def compute_binding_indicies(exp_mu,  # Expected marginal utility
                             c_bind,  # Current guess of c_bind
                             model, tol=1e-7): 

    # Calculate utility differential when consumption is constrained
    mu_constrained = compute_marginal_utility(c_bind, model)
    constrained_euler_diff = mu_constrained - exp_mu

    # Indices where the constraints bind
    return np.where(constrained_euler_diff > tol, 1, 0)


def compute_consumption_at_constraint(c, model):

    # Unpack and set up
    σ, η, β, ω, κ, R, b_grid, yn_grid, yt_grid, P = model
    n_B, n_y =  len(b_grid), len(yn_grid)
    b_grid_min, b_grid_max = b_grid[0], b_grid[-1]

    # Reshape
    b = np.reshape(b_grid, (n_B, 1))
    yt = np.reshape(yt_grid, (1, n_y))
    yn = np.reshape(yn_grid, (1, n_y))

    # Get current price
    price = (1 - ω) / ω * (c / yn)**(1 + η)

    # Obtain bond purchases at the constraint
    b_bind = - κ * (price * yn + yt)
    b_bind[b_bind > b_grid_max] = b_grid_max
    b_bind[b_bind < b_grid_min] = b_grid_min
    
    # Update c_bind
    c_bind = R * b + yt - b_bind
    c_bind[c_bind < 0] = np.inf

    return c_bind


def decentralized_update(c,          # Current consumption policy
                         c_bind,     # Consumption at constraint
                         bp,         # Current bond purchase policy
                         model):     # Instance of Model

    # Unpack and set up
    σ, η, β, ω, κ, R, b_grid, yn_grid, yt_grid, P = model
    b_grid_min, b_grid_max = b_grid[0], b_grid[-1]
    n_B =  len(b_grid)
    n_y =  len(yn_grid)

    # Reshape
    b = np.reshape(b_grid, (n_B, 1))
    yt = np.reshape(yt_grid, (1, n_y))
    yn = np.reshape(yn_grid, (1, n_y))

    # Make c a new array so it won't be modified in place
    old_c = c
    c = np.empty_like(old_c)

    # Compute expected marginal utility given current guess of c, bp
    mu = compute_marginal_utility(old_c, model)
    exp_mu = compute_exp_marginal_utility(mu, bp, model)

    # Indices where the constraints bind
    idx_bind = compute_binding_indicies(exp_mu, c_bind, model)

    # Update consumption 
    for i in range(n_B):
        for j in range(n_y):

            if idx_bind[i, j]:           # Use borrowing constraint to set c
                c[i, j] = c_bind[i, j]
            else:                        # Use Euler equation to find c
                def euler_diff(cc):
                    return (ω*cc**(-η) + (1-ω)*yn_grid[j]**(-η))**(σ/η -1/η -1) \
                                * ω*cc**(-η-1) - exp_mu[i, j]
                c0 = old_c[i, j]
                c[i, j] = newton(euler_diff, c0)  

    # Update bp   
    bp = R * b + yt - c
    bp[bp > b_grid_max] = b_grid_max
    bp[bp < b_grid_min] = b_grid_min
    price = (1 - ω) / ω * (c / yn)**(1 + η)
    c = R * b + yt - np.maximum(bp, -κ * (price * yn + yt))

    # Update c_bind based on the new c
    c_bind = compute_consumption_at_constraint(c, model)

    return c, c_bind, bp, idx_bind


def solve_decentralized_equilibrium(model,              # Instance of Model
                                    α=0.2,              # Damping parameter
                                    print_step=10,      # Display frequency
                                    iter_max=50_000,    # Iteration tolerance
                                    tol=1.0e-5):        # Numerical tolerance

     
    # Initialize 
    c, c_bind, bp = initialize_decentralized_eq_search(model)
    current_iter = 0
    error = tol + 1

    while error > tol and current_iter < iter_max:
    
        # Store current values and update
        old_c  = c
        old_bp = bp
        updates = decentralized_update(c, c_bind, bp, model)
        c, c_bind, bp, idx_bind = updates

        # Compute error and update iteration
        error = max(d_infty(c, old_c), d_infty(bp, old_bp))
        current_iter += 1
        if current_iter % print_step == 0:
            print(f"Current error = {error} at iteration {current_iter}.")
    
        # Add smoothing 
        bp = α*bp + (1-α)*old_bp
        c = α*c + (1-α)*old_c

    print(f"DE converged at iteration {current_iter}.")
    return c, bp

# Solve the model
model = create_overborrowing_model()
print("Starting solver\n")
c, bp = solve_decentralized_equilibrium(model)
print("Solver done\n")
b_grid = model.b_grid

# Here's a plot but this doesn't quite line up with the first figure created by main.m

# Plot
fig, ax = plt.subplots()
ax.plot(b_grid, b_grid, 'k--')
ax.plot(b_grid, bp[:, 4])
plt.show()




