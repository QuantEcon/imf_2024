---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Wealth Dynamics

In this lecture we examine wealth dynamics in large cross-section of agents who
are subject to both 

* idiosyncratic shocks, which affect labor income and returns, and 
* aggregate shocks, which also impact on labor income and returns

Savings and consumption behavior is taken as given -- you can plug in your
favorite model and then analyze distribution dynamics using these techniques.

Uncomment if necessary

```{code-cell} ipython3
#!pip install quantecon 
```

We use the following imports.

```{code-cell} ipython3
import numba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import jax
import jax.numpy as jnp
from collections import namedtuple
```


## Wealth dynamics: Numba version


The model we will study is

```{math}
    w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

where

- $w_t$ is wealth at time $t$ for a given household,
- $r_t$ is the rate of return of financial assets,
- $y_t$ is labor income and
- $s(w_t)$ is savings (current wealth minus current consumption)

In addition, there is an aggregate state process $\{z_t\}$ obeying

$$
    z_{t+1} = a z_t + b + \sigma_z \epsilon_{t+1}
$$

This aggregate process affects the interest rate and labor income.

we’ll assume that

$$
    R_t := 1 + r_t = c_r \exp(z_t) + \exp(\mu_r + \sigma_r \xi_t)
$$

and

$$
    y_t = c_y \exp(z_t) + \exp(\mu_y + \sigma_y \zeta_t)
$$

Here $\{ (\epsilon_t, \xi_t, \zeta_t) \}$ is IID and standard normal in $\mathbb R^3$.

The value of $c_r$ should be close to zero, since rates of return on assets do not exhibit large trends.

When we simulate a population of households, we will assume all shocks are idiosyncratic (i.e.,  specific to individual households and independent across them).

Regarding the savings function $s$, our default model will be

```{math}
:label: sav_ah

s(w) = s_0 w \cdot \mathbb 1\{w \geq \hat w\}
```

where $s_0$ is a positive constant.

Thus, for $w < \hat w$, the household saves nothing. For $w \geq \bar w$, the household saves a fraction $s_0$ of their wealth.

We are using something akin to a fixed savings rate model, while acknowledging that low wealth households tend to save very little.

In most macroeconomic models the savings function will be determined by
optimization.

We abstract from this step --- consider this analysis of distribution dynamics
with when a savings function has already been determined.


## Implementation


Here's a function that collects parameters and useful constants

```{code-cell} ipython3
def create_wealth_model(w_hat=1.0,   # Savings parameter
                        s_0=0.75,    # Savings parameter
                        c_y=1.0,     # Labor income parameter
                        μ_y=1.0,     # Labor income parameter
                        σ_y=0.2,     # Labor income parameter
                        c_r=0.05,    # Rate of return parameter
                        μ_r=0.1,     # Rate of return parameter
                        σ_r=0.5,     # Rate of return parameter
                        a=0.5,       # Aggregate shock parameter
                        b=0.0,       # Aggregate shock parameter
                        σ_z=0.1):    # Aggregate shock parameter
    """
    Create a wealth model with given parameters. 
    
    """
    # Mean and variance of z process
    z_mean = b / (1 - a)
    z_var = σ_z**2 / (1 - a**2)
    exp_z_mean = np.exp(z_mean + z_var / 2)
    # Mean of R and y processes
    R_mean = c_r * exp_z_mean + np.exp(μ_r + σ_r**2 / 2)
    y_mean = c_y * exp_z_mean + np.exp(μ_y + σ_y**2 / 2)
    # Test stability condition ensuring wealth does not diverge
    # to infinity.
    α = R_mean * s_0
    if α >= 1:
        raise ValueError("Stability condition failed.")
    # Pack values into tuples and return them
    household_params = (w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, y_mean)
    aggregate_params = (a, b, σ_z, z_mean, z_var)
    model = household_params, aggregate_params
    return model
```

```{code-cell} ipython3
@numba.jit
def generate_aggregate_state_sequence(aggregate_params, length=100):
    a, b, σ_z, z_mean, z_var = aggregate_params 
    z = np.empty(length+1)
    z[0] = z_mean   # Initialize at z_mean
    for t in range(length):
        z[t+1] = a * z[t] + b + σ_z * np.random.randn()
    return z
```

Here's two functions that update the aggregate state and household wealth.

```{code-cell} ipython3
@numba.jit
def update_wealth(household_params, w, z):
    """
    Generate w_{t+1} given w_t and z_{t+1}.
    """
    # Unpack
    w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, y_mean = household_params
    # Update wealth
    y = c_y * np.exp(z) + np.exp(μ_y + σ_y * np.random.randn())
    wp = y
    if w >= w_hat:
        R = c_r * np.exp(z) + np.exp(μ_r + σ_r * np.random.randn())
        wp += R * s_0 * w
    return wp
```

Here's function to simulate the time series of wealth for individual households.

```{code-cell} ipython3
@numba.jit
def wealth_time_series(model, w_0, n):
    """
    Generate a single time series of length n for wealth given
    initial value w_0.

    The initial persistent state z_0 for each household is drawn from
    the stationary distribution of the AR(1) process.

    """
    # Unpack
    household_params, aggregate_params = model
    a, b, σ_z, z_mean, z_var = aggregate_params 
    # Initialize and update
    z = generate_aggregate_state_sequence(aggregate_params, length=n)
    w = np.empty(n+1)
    w[0] = w_0
    for t in range(n):
        w[t+1] = update_wealth(household_params, w[t], z[t+1])
    return w
```

Here's function to simulate a cross section of households forward in time.

Note the use of parallelization to speed up computation.

```{code-cell} ipython3
@numba.jit(parallel=True)
def update_cross_section(model, w_distribution, shift_length=500):
    """
    Shifts a cross-section of household forward in time

    Takes a current distribution of wealth values as w_distribution
    and updates each w_t in w_distribution to w_{t+j}, where
    j = shift_length.

    Returns the new distribution.

    """
    # Unpack
    household_params, aggregate_params = model

    num_households = len(w_distribution)
    new_distribution = np.empty_like(w_distribution)
    z = generate_aggregate_state_sequence(aggregate_params,
                                          length=shift_length)

    # Update each household
    for i in numba.prange(num_households):
        # 
        w = w_distribution[i]
        for t in range(shift_length-1):
            w = update_wealth(household_params, w, z[t])
        new_distribution[i] = w
    return new_distribution
```

Parallelization is very effective in the function above because the time path
of each household can be calculated independently once the path for the
aggregate state is known.

## Applications

Let's try simulating the model at different parameter values and investigate
the implications for the wealth distribution.

### Time Series

Let's look at the wealth dynamics of an individual household.

```{code-cell} ipython3
model = create_wealth_model()
household_params, aggregate_params = model
w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, y_mean = household_params
a, b, σ_z, z_mean, z_var = aggregate_params 
ts_length = 200
w = wealth_time_series(model, y_mean, ts_length)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w)
plt.show()
```

Notice the large spikes in wealth over time.

Such spikes are similar to what we observed in time series when {doc}`we studied Kesten processes <kesten_processes>`.

### Inequality Measures

Let's look at how inequality varies with returns on financial assets.

The next function generates a cross section and then computes the Lorenz
curve and Gini coefficient.

```{code-cell} ipython3
def generate_lorenz_and_gini(model, num_households=100_000, T=500):
    """
    Generate the Lorenz curve data and Gini coefficient by simulating
    num_households forward to time T.
    """
    # Unpack
    household_params, aggregate_params = model
    a, b, σ_z, z_mean, z_var = aggregate_params 
    # Initialize
    ψ_0 = np.full(num_households, y_mean)
    z_0 = z_mean
    # Compute cross-section and measures
    ψ_star = update_cross_section(model, ψ_0, shift_length=T)
    return qe.gini_coefficient(ψ_star), qe.lorenz_curve(ψ_star)
```

Now we investigate how the Lorenz curves associated with the wealth distribution change as return to savings varies.

The code below plots Lorenz curves for three different values of $\mu_r$.

If you are running this yourself, note that it will take one or two minutes to execute.

This is unavoidable because we are executing a CPU intensive task.

In fact the code, which is JIT compiled and parallelized, runs extremely fast relative to the number of computations.

```{code-cell} ipython3
%%time

fig, ax = plt.subplots()
μ_r_vals = (0.0, 0.025, 0.05)
gini_vals = []

for μ_r in μ_r_vals:
    model = create_wealth_model(μ_r=μ_r)
    gv, (f_vals, l_vals) = generate_lorenz_and_gini(model)
    ax.plot(f_vals, l_vals, label=f'$\psi^*$ at $\mu_r = {μ_r:0.2}$')
    gini_vals.append(gv)

ax.plot(f_vals, f_vals, label='equality')
ax.legend(loc="upper left")
plt.show()
```

The Lorenz curve shifts downwards as returns on financial income rise, indicating a rise in inequality.

We will look at this again via the Gini coefficient immediately below, but
first consider the following image of our system resources when the code above
is executing:

Since the code is both efficiently JIT compiled and fully parallelized, it's
close to impossible to make this sequence of tasks run faster without changing
hardware.

Now let's check the Gini coefficient.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(μ_r_vals, gini_vals, label='gini coefficient')
ax.set_xlabel("$\mu_r$")
ax.legend()
plt.show()
```

Once again, we see that inequality increases as returns on financial income
rise.

Let's finish this section by investigating what happens when we change the
volatility term $\sigma_r$ in financial returns.

```{code-cell} ipython3
%%time

fig, ax = plt.subplots()
σ_r_vals = (0.35, 0.45, 0.52)
gini_vals = []

for σ_r in σ_r_vals:
    model = create_wealth_model(σ_r=σ_r)
    gv, (f_vals, l_vals) = generate_lorenz_and_gini(model)
    ax.plot(f_vals, l_vals, label=f'$\psi^*$ at $\sigma_r = {σ_r:0.2}$')
    gini_vals.append(gv)

ax.plot(f_vals, f_vals, label='equality')
ax.legend(loc="upper left")
plt.show()
```

### Pareto tails

In most countries, the cross-sectional distribution of wealth exhibits a Pareto
tail (power law).


Let's see if our model can replicate this stylized fact by running a simulation
that generates a cross-section of wealth and generating a suitable rank-size plot.

We will use the function `rank_size` from `quantecon` library.

In the limit, data that obeys a power law generates a straight line.

```{code-cell} ipython3
model = create_wealth_model()
household_params, aggregate_params = model
w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, y_mean = household_params
a, b, σ_z, z_mean, z_var = aggregate_params 

num_households = 250_000
T = 500                                      # shift forward T periods
ψ_0 = np.full(num_households, y_mean)   # initial distribution
z_0 = z_mean
```

First let's generate the distribution:

```{code-cell} ipython3
num_households = 250_000
T = 500  # how far to shift forward in time
model = create_wealth_model()
ψ_0 = np.full(num_households, y_mean)
z_0 = z_mean

ψ_star = update_cross_section(model, ψ_0, shift_length=T)
```

Now let's see the rank-size plot:

```{code-cell} ipython3
fig, ax = plt.subplots()

rank_data, size_data = qe.rank_size(ψ_star, c=0.001)
ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
ax.set_xlabel("log rank")
ax.set_ylabel("log size")

plt.show()
```

## JAX version

First we define a function that implements the Lorenz curve.

```{code-cell} ipython3
@jax.jit
def lorenz_curve_jax(y):
    n = y.shape[0]
    y = jnp.sort(y)
    s = jnp.concatenate((jnp.zeros(1), jnp.cumsum(y)))
    _cum_p = jnp.arange(1, n + 1) / n
    cum_income = s / s[n]
    cum_people = jnp.concatenate((jnp.zeros(1), _cum_p))
    return cum_people, cum_income
```

Here's a function that computes the Gini coefficient.

```{code-cell} ipython3
@jax.jit
def gini_jax(y):
    n = y.shape[0]
    g_sum = 0

    def sum_y_gini(i, g_sum):
        g_sum += jnp.sum(jnp.abs(y[i] - y))
        return g_sum
    
    g_sum = jax.lax.fori_loop(0, n, sum_y_gini, 0)
    return g_sum / (2 * n * jnp.sum(y))
```


## Wealth dynamics using JAX

Let's define a model to represent the wealth dynamics.


We'll organize the data in a different way in order to simplify the work of the
JAX compiler.

```{code-cell} ipython3
# NamedTuple Model

Model = namedtuple("Model", ("w_hat", "s_0", "c_y", "μ_y",
                             "σ_y", "c_r", "μ_r", "σ_r", "a",
                             "b", "σ_z", "z_mean", "z_var", "y_mean"))
```

Here's a function to create an instance with the same parameters as above.

```{code-cell} ipython3
def create_wealth_model(w_hat=1.0,
                        s_0=0.75,
                        c_y=1.0,
                        μ_y=1.0,
                        σ_y=0.2,
                        c_r=0.05,
                        μ_r=0.1,
                        σ_r=0.5,
                        a=0.5,
                        b=0.0,
                        σ_z=0.1):
    """
    Create a wealth model with given parameters and return
    and instance of NamedTuple Model.
    """
    z_mean = b / (1 - a)
    z_var = σ_z**2 / (1 - a**2)
    exp_z_mean = jnp.exp(z_mean + z_var / 2)
    R_mean = c_r * exp_z_mean + jnp.exp(μ_r + σ_r**2 / 2)
    y_mean = c_y * exp_z_mean + jnp.exp(μ_y + σ_y**2 / 2)
    # Test a stability condition that ensures wealth does not diverge
    # to infinity.
    α = R_mean * s_0
    if α >= 1:
        raise ValueError("Stability condition failed.")
    return Model(w_hat=w_hat, s_0=s_0, c_y=c_y, μ_y=μ_y,
                 σ_y=σ_y, c_r=c_r, μ_r=μ_r, σ_r=σ_r, a=a,
                 b=b, σ_z=σ_z, z_mean=z_mean, z_var=z_var, y_mean=y_mean)
```

The following function updates one period with the given current wealth and persistent state.

```{code-cell} ipython3
def update_states_jax(arrays, model, size, rand_key):
    """
    Update one period, given current wealth w and persistent
    state z. They are stored in the form of tuples under the arrays argument
    """
    # Unpack w and z
    w, z = arrays

    rand_key, *subkey = jax.random.split(rand_key, 3)
    zp = a * z + b + σ_z * jax.random.normal(rand_key, shape=size)

    # Update wealth
    y = c_y * jnp.exp(zp) + jnp.exp(
                        μ_y + σ_y * jax.random.normal(subkey[0], shape=size))
    wp = y

    R = c_r * jnp.exp(zp) + jnp.exp(
                        μ_r + σ_r * jax.random.normal(subkey[1], shape=size))
    wp += (w >= w_hat) * R * s_0 * w
    return wp, zp
```

Here’s function to simulate the time series of wealth for individual households using a `for` loop and JAX.

```{code-cell} ipython3
# Using JAX and for loop

def wealth_time_series_for_loop_jax(w_0, n, model, size, rand_seed=1):
    """
    Generate a single time series of length n for wealth given
    initial value w_0.

    * This implementation uses a `for` loop.

    The initial persistent state z_0 for each household is drawn from
    the stationary distribution of the AR(1) process.

        * model: NamedTuple Model
        * w_0: scalar/vector
        * n: int
        * size: size/shape of the w_0
        * rand_seed: int (Used to generate PRNG key)
    """
    rand_key = jax.random.PRNGKey(rand_seed)
    rand_key, *subkey = jax.random.split(rand_key, n)

    w_0 = jax.device_put(w_0).reshape(size)

    z = z_mean + jnp.sqrt(z_var) * jax.random.normal(rand_key, shape=size)
    w = [w_0]
    for t in range(n-1):
        w_, z = update_states_jax((w[t], z), model, size, subkey[t])
        w.append(w_)
    return jnp.array(w)
```

Let's try simulating the model at different parameter values and investigate the implications for the wealth distribution using the above function.

```{code-cell} ipython3
model = create_wealth_model() # default model
ts_length = 200
size = (1,)
```

```{code-cell} ipython3
%%time

w_jax_result = wealth_time_series_for_loop_jax(y_mean,
                                               ts_length, model, size).block_until_ready()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_jax_result)
plt.show()
```

We can further try to optimize and speed up the compile time of the above function by replacing `for` loop with [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html).

```{code-cell} ipython3
def wealth_time_series_jax(w_0, n, model, size, rand_seed=1):
    """
    Generate a single time series of length n for wealth given
    initial value w_0.

    * This implementation uses `jax.lax.scan`.

    The initial persistent state z_0 for each household is drawn from
    the stationary distribution of the AR(1) process.

        * model: NamedTuple Model
        * w_0: scalar/vector
        * n: int
        * size: size/shape of the w_0
        * rand_seed: int (Used to generate PRNG key)
    """
    rand_key = jax.random.PRNGKey(rand_seed)
    rand_key, *subkey = jax.random.split(rand_key, n)

    w_0 = jax.device_put(w_0).reshape(size)
    z_init = z_mean + jnp.sqrt(z_var) * jax.random.normal(rand_key, shape=size)
    arrays = w_0, z_init
    rand_sub_keys = jnp.array(subkey)

    w_final = jnp.array([w_0])

    # Define the function for each update
    def update_w_z(arrays, rand_sub_key):
        wp, zp = update_states_jax(arrays, model, size, rand_sub_key)
        return (wp, zp), wp

    arrays_last, w_values = jax.lax.scan(update_w_z, arrays, rand_sub_keys)
    return jnp.concatenate((w_final, w_values))

# Create the jit function
wealth_time_series_jax = jax.jit(wealth_time_series_jax, static_argnums=(1,3,))
```

Let's try simulating the model at different parameter values and investigate the implications for the wealth distribution and also observe the difference in time between `wealth_time_series_jax` and `wealth_time_series_for_loop_jax`.

```{code-cell} ipython3
model = create_wealth_model() # default model
ts_length = 200
size = (1,)
```

```{code-cell} ipython3
%%time

w_jax_result = wealth_time_series_jax(y_mean, ts_length, model, size).block_until_ready()
```

Running the above function again will be even faster because of JAX's JIT.

```{code-cell} ipython3
%%time

# 2nd time is expected to be very fast because of JIT
w_jax_result = wealth_time_series_jax(y_mean, ts_length, model, size).block_until_ready()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_jax_result)
plt.show()
```

Now here’s function to simulate a cross section of households forward in time.

```{code-cell} ipython3
def update_cross_section_jax(w_distribution, shift_length, model, size, rand_seed=2):
    """
    Shifts a cross-section of household forward in time

    * model: NamedTuple Model
    * w_distribution: array_like, represents current cross-section

    Takes a current distribution of wealth values as w_distribution
    and updates each w_t in w_distribution to w_{t+j}, where
    j = shift_length.

    Returns the new distribution.
    """
    new_dist = wealth_time_series_jax(w_distribution, shift_length, model, size, rand_seed)
    new_distribution = new_dist[-1, :]
    return new_distribution


# Create the jit function
update_cross_section_jax = jax.jit(update_cross_section_jax, static_argnums=(1,3,))
```

## Applications

Let's try simulating the model at different parameter values and investigate
the implications for the wealth distribution.


### Inequality Measures

Let's look at how inequality varies with returns on financial assets.

The next function generates a cross section and then computes the Lorenz
curve and Gini coefficient.

```{code-cell} ipython3
def generate_lorenz_and_gini_jax(model, num_households=100_000, T=500):
    """
    Generate the Lorenz curve data and Gini coefficient 
    by simulating num_households forward to time T.
    """
    size = (num_households, )
    ψ_0 = jnp.full(size, y_mean)
    ψ_star = update_cross_section_jax(ψ_0, T, model, size)
    return gini_jax(ψ_star), lorenz_curve_jax(ψ_star)

# Create the jit function
generate_lorenz_and_gini_jax = jax.jit(generate_lorenz_and_gini_jax,
                                       static_argnums=(1,2,))
```

Now we investigate how the Lorenz curves associated with the wealth distribution change as return to savings varies.

The code below plots Lorenz curves for three different values of $\mu_r$.

```{code-cell} ipython3
%%time

fig, ax = plt.subplots()
μ_r_vals = (0.0, 0.025, 0.05)
gini_vals = []

for μ_r in μ_r_vals:
    model = create_wealth_model(μ_r=μ_r)
    gv, (f_vals, l_vals) = generate_lorenz_and_gini_jax(model)
    ax.plot(f_vals, l_vals, label=f'$\psi^*$ at $\mu_r = {μ_r:0.2}$')
    gini_vals.append(gv)

ax.plot(f_vals, f_vals, label='equality')
ax.legend(loc="upper left")
plt.show()
```

The Lorenz curve shifts downwards as returns on financial income rise, indicating a rise in inequality.

Now let's check the Gini coefficient.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(μ_r_vals, gini_vals, label='gini coefficient')
ax.set_xlabel("$\mu_r$")
ax.legend()
plt.show()
```

Once again, we see that inequality increases as returns on financial income
rise.

Let's finish this section by investigating what happens when we change the
volatility term $\sigma_r$ in financial returns.

```{code-cell} ipython3
%%time

fig, ax = plt.subplots()
σ_r_vals = (0.35, 0.45, 0.52)
gini_vals = []

for σ_r in σ_r_vals:
    model = create_wealth_model(σ_r=σ_r)
    gv, (f_vals, l_vals) = generate_lorenz_and_gini_jax(model)
    ax.plot(f_vals, l_vals, label=f'$\psi^*$ at $\sigma_r = {σ_r:0.2}$')
    gini_vals.append(gv)

ax.plot(f_vals, f_vals, label='equality')
ax.legend(loc="upper left")
plt.show()
```

We see that greater volatility has the effect of increasing inequality in this model.
