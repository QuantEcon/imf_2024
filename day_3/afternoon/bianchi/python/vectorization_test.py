import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from mc_dynamics import discretize_income_var


def create_overborrowing_model(
        σ=2,                 # CRRA utility parameter
        η=(1/0.83)-1,        # Elasticity = 0.83, η = 0.2048
        β=0.91,              # Discount factor
        ω=0.31,              # Share for tradables
        κ=0.3235,            # Constraint parameter
        r=0.04,              # Interest rate
        b_grid_size=2,       # Bond grid size, increase to 800
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
    y_t_nodes, y_n_nodes, Q = discretize_income_var(grid_size=2, 
                                                    single_index=False)
    # Set up grid for bond holdings
    b_grid = np.linspace(b_grid_min, b_grid_max, b_grid_size)

    parameters = σ, η, β, ω, κ, r
    sizes = b_grid_size, len(Q)
    # This line is for later: switch np to jnp 
    arrays = tuple(map(np.array, (b_grid, y_t_nodes, y_n_nodes, Q)))
    return parameters, sizes, arrays



def w(model, c, y_n):
    """ 
    Current utility when c_t = c and c_n = y_n.

        a = [ω c^(- η) + (1 - ω) y_n^(- η)]^(-1/η)

        w(c, y_n) := a^(1 - σ) / (1 - σ)

    """
    parameters, sizes, arrays = model
    σ, η, β, ω, κ, r = parameters
    a = (ω * c**(-η) + (1 - ω) * y_n**(-η))**(-1/η)
    return a**(1 - σ) / (1 - σ)


def loop_generate_initial_H(model):
    """
    Compute an initial guess for H. Use a hold-steady rule.

    """
    parameters, sizes, arrays = model
    b_size, y_size = sizes
    b_grid, y_t_nodes, y_n_nodes, Q = arrays

    H = np.empty((b_size, y_size, y_size))
    for i_B in range(b_size):
        for i_y_t in range(y_size):
            for i_y_n in range(y_size):
                H[i_B, i_y_t, i_y_n] = b_grid[i_B]  # hold steady rule
    return H


def generate_initial_H(model):
    """
    Compute an initial guess for H. Use a hold-steady rule, where

        H[B, y_t, y_n] = B  for all y_t, y_n

    """
    parameters, sizes, arrays = model
    b_size, y_size = sizes
    b_grid, y_t_nodes, y_n_nodes, Q = arrays
    b_grid = np.reshape(b_grid, (b_size, 1, 1)) # b' = b
    return b_grid * np.ones((b_size, y_size, y_size))


def loop_T(model, v, H):
    """
    Bellman operator.  The value function indices have the form

        v = v[i_b, i_B, i_y_t, i_y_n]

    """
    parameters, sizes, arrays = model
    σ, η, β, ω, κ, r = parameters
    b_size, y_size = sizes
    b_grid, y_t_nodes, y_n_nodes, Q = arrays

    # Storage for new value function
    v_new = np.empty_like(v)
    # Storage for greedy policy
    bp_v_greedy = np.empty_like(v)
    
    for i_y_t in range(y_size):
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
                    bp_maximizer = 0.0
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

    return v_new #, bp_v_greedy

def loop_EV(model, v, H):
    """
    Bellman operator.  The value function indices have the form

        v = v[i_b, i_B, i_y_t, i_y_n]

    """
    parameters, sizes, arrays = model
    σ, η, β, ω, κ, r = parameters
    b_size, y_size = sizes
    b_grid, y_t_nodes, y_n_nodes, Q = arrays

    # Storage for new value function
    v_new = np.empty_like(v)
    # Storage for greedy policy
    bp_v_greedy = np.empty_like(v)
    
    for i_y_t in range(y_size):
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
                    bp_maximizer = 0.0
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

    return v_new #, bp_v_greedy


def T(model, v, H):
    """
    The Bellman operator.

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
    σ, η, β, ω, κ, r = parameters
    b_size, y_size = sizes
    b_grid, y_t_nodes, y_n_nodes, Q = arrays

    # Step one: Compute the expectation of the value function.
    # We expand the arrays out to shape 
    # 
    #  (b, B, y_t, y_n, b', B', y_tp, y_np)  
    #
    # and then replace B' with H(B, y_t, y_n) and sum out y_tp, y_np, 
    # multiplying by the Markov matrix Q.


    # Expand dimension of arrays
    b   = np.reshape(b_grid,    (b_size, 1, 1, 1, 1, 1, 1, 1))
    B   = np.reshape(b_grid,    (1, b_size, 1, 1, 1, 1, 1, 1))
    y_t = np.reshape(y_t_nodes, (1, 1, y_size, 1, 1, 1, 1, 1))
    y_n = np.reshape(y_n_nodes, (1, 1, 1, y_size, 1, 1, 1, 1))
    bp  = np.reshape(b_grid,    (1, 1, 1, 1, b_size, 1, 1, 1))
    Bp  = np.reshape(H, 
                      (1, b_size, y_size, y_size, 1, 1, 1, 1))
    y_tp = np.reshape(y_t_nodes, (1, 1, 1, 1, 1, 1, y_size, 1))
    y_np = np.reshape(y_n_nodes, (1, 1, 1, 1, 1, 1, 1, y_size))

    # Provide some index arrays of the same shape
    b_idx   = np.reshape(range(b_size), 
                                 (b_size, 1, 1, 1, 1, 1, 1, 1))
    bp_idx  = np.reshape(range(b_size), 
                                 (1, b_size, 1, 1, 1, 1, 1, 1))
    y_t_idx = np.reshape(range(y_size), 
                                 (1, 1, y_size, 1, 1, 1, 1, 1))
    y_n_idx = np.reshape(range(y_size), 
                                 (1, 1, 1, y_size, 1, 1, 1, 1))
    bp_idx  = np.reshape(range(b_size), 
                                 (1, 1, 1, 1, b_size, 1, 1, 1))
    Bp_idx =  np.searchsorted(b_grid, Bp) 
    y_tp_idx = np.reshape(range(y_size), 
                                 (1, 1, 1, 1, 1, 1, y_size, 1))
    y_np_idx = np.reshape(range(y_size), 
                                 (1, 1, 1, 1, 1, 1, 1, y_size))

    V = np.reshape(Q, (1, 1, 1, 1, b_size, b_size, y_size, y_size))
    Q = np.reshape(Q, (1, 1, y_size, y_size, 1, 1, y_size, y_size))
    EV = np.sum(V * Q * B_idx, axes=(5, 6, 7))

    return EV

    # compute price of nontradables using aggregates
    #C = (1 + r) * B + y_t - Bp
    #P = ((1 - ω) / ω) * (C / y_n)**(η + 1)

    #c = (1 + r) * b + y_t - bp
    #u = w(model, c, y_n)

    #constraint_holds = - κ * (P * y_n + y_t) <= bp <= (1 + r) * b + y_t

    # Q[y_t, y_n, y_tp, y_np] -> Q[b, B, y_t, y_n, bp, Bb, y_tp, y_np]
    # v[bp, Bb, y_tp, y_np]   -> v[b, B, y_t, y_n, bp, Bb, y_tp, y_np]
    #v = np.resize(v, ?)
    #EV = np.sum(v * Q, axis=?)

    #W = np.where(constraint_holds, u + β * EV, -np.inf)
    #v_new       = np.max(W, axis=?)
    #bp_v_greedy = np.argmax(W, axis=?)

    #return v_new, bp_v_greedy


model = create_overborrowing_model()
parameters, sizes, arrays = model
σ, η, β, ω, κ, r = parameters
b_size, y_size = sizes
b_grid, y_t_nodes, y_n_nodes, Q = arrays
v = np.ones((b_size, b_size, y_size, y_size))

H = loop_generate_initial_H(model)
Tv = loop_T(model, v, H)
#H = generate_initial_H(model)
#Tv_loop = loop_T(model, v, H)


