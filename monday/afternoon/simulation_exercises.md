---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Simulation Exercises

#### Prepared for the IMF Computational Workshop (March 2024)

#### Chase Coleman and John Stachurski

+++

This notebook contains some exercises related to simulation.  

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit, prange
```


## Exercise

Compute an approximation to $ \pi $ using [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method).

Your hints are as follows:

- If $ U $ is a bivariate uniform random variable on the unit square $ (0, 1)^2 $, then the probability that $ U $ lies in a subset $ B $ of $ (0,1)^2 $ is equal to the area of $ B $.  
- If $ U_1,\ldots,U_n $ are IID copies of $ U $, then, as $ n $ gets large, the fraction that falls in $ B $, converges to the probability of landing in $ B $.  
- For a circle, $ area = \pi * radius^2 $.  

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

Consider the circle of diameter 1 embedded in the unit square.

Let $ A $ be its area and let $ r=1/2 $ be its radius, so that $A = \pi r^2 $.

If we can estimate $A$ then we can estimate $ \pi $ via $ \pi = A / r^2 = 4A$.

We estimate $A$ by sampling bivariate uniforms and looking at the fraction that falls into the circle.

```{code-cell} ipython3
n = 1_000_000 # sample size for Monte Carlo simulation

def in_circle(u, v):
    """
    Test whether (u, v) falls within the unit circle centred at (0.5,0.5)
    """
    d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)
    return d < 0.5

count = 0
for i in range(n):

    # drawing random positions on the square
    u, v = np.random.uniform(0, 1), np.random.uniform(0, 1)

    # if it falls within the circle, add it to the count
    if in_circle(u, v):
        count += 1

area_estimate = count / n

print(area_estimate * 4)  # dividing by radius**2
```


## Exercise

Accelerate the code from the previous exercise using Numba.  Time the difference.

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

```{code-cell} ipython3
def calculate_pi(n=1_000_000):
    count = 0
    for i in range(n):
        u, v = np.random.uniform(0, 1), np.random.uniform(0, 1)
        d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)
        if d < 0.5:
            count += 1
    area_estimate = count / n
    return area_estimate * 4  # dividing by radius**2
```

```{code-cell} ipython3
%time calculate_pi()
```

```{code-cell} ipython3
fast_calc_pi = jit(calculate_pi)
```

```{code-cell} ipython3
%time fast_calc_pi()
```

And again to omit compile time:

```{code-cell} ipython3
%time fast_calc_pi()
```

## Exercise

Suppose that the volatility of returns on an asset can be in one of two regimes — high or low.

The transition probabilities across states are as follows

![https://python-programming.quantecon.org/_static/lecture_specific/sci_libs/nfs_ex1.png](https://python-programming.quantecon.org/_static/lecture_specific/sci_libs/nfs_ex1.png)

  
For example, let the period length be one day, and suppose the current state is high.

We see from the graph that the state tomorrow will be

- high with probability 0.8  
- low with probability 0.2  


Your task is to simulate a sequence of daily volatility states according to this rule.

Set the length of the sequence to `n = 1_000_000` and start in the high state.

Implement a pure Python version and a Numba version, and compare speeds.

To test your code, evaluate the fraction of time that the chain spends in the low state.

If your code is correct, it should be about 2/3.

Hints:

- Represent the low state as 0 and the high state as 1.  
- If you want to store integers in a NumPy array and then apply JIT compilation, use `x = np.empty(n, dtype=numba.int64)` or similar.  

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

We let

- 0 represent “low”  
- 1 represent “high”  

```{code-cell} ipython3
p, q = 0.1, 0.2  # Prob of leaving low and high state respectively
```

Here’s a pure Python version of the function

```{code-cell} ipython3
def compute_series(n):
    x = np.empty(n, dtype=int)
    x[0] = 1  # Start in state 1
    U = np.random.uniform(0, 1, size=n)
    for t in range(1, n):
        current_x = x[t-1]
        if current_x == 0:
            x[t] = U[t] < p
        else:
            x[t] = U[t] > q
    return x
```

```{code-cell} ipython3
n = 1_000_000
```

```{code-cell} ipython3
%time x = compute_series(n)
```

```{code-cell} ipython3
print(np.mean(x == 0))  # Fraction of time x is in state 0
```

Now let's speed it up:

```{code-cell} ipython3

```{code-cell} ipython3
@jit
def fast_compute_series(n):
    x = np.empty(n, dtype=numba.int8)
    x[0] = 1  # Start in state 1
    U = np.random.uniform(0, 1, size=n)
    for t in range(1, n):
        current_x = x[t-1]
        if current_x == 0:
            x[t] = U[t] < p
        else:
            x[t] = U[t] > q
    return x
```

```

Run once to compile:

```{code-cell} ipython3
%time fast_compute_series(n)
```

Now let's check the speed:

```{code-cell} ipython3
%time fast_compute_series(n)
```


**Exercise**


We consider using Monte Carlo to price a European call option.

The price of the option obeys 

$$
P = \beta^n \mathbb E \max\{ S_n - K, 0 \}
$$

where

1. $\beta$ is a discount factor,
2. $n$ is the expiry date,
2. $K$ is the strike price and
3. $\{S_t\}$ is the price of the underlying asset at each time $t$.

Suppose that `n, β, K = 20, 0.99, 100`.

Assume that the stock price obeys 

$$ 
\ln \frac{S_{t+1}}{S_t} = \mu + \sigma_t \xi_{t+1}
$$

where 

$$ 
    \sigma_t = \exp(h_t), 
    \quad
        h_{t+1} = \rho h_t + \nu \eta_{t+1}
$$

Here $\{\xi_t\}$ and $\{\eta_t\}$ are IID and standard normal.

(This is a stochastic volatility model, where the volatility $\sigma_t$ varies over time.)

Use the defaults `μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0`.

(Here `S0` is $S_0$ and `h0` is $h_0$.)

By generating $M$ paths $s_0, \ldots, s_n$, compute the Monte Carlo estimate 

$$
    \hat P_M 
    := \beta^n \mathbb E \max\{ S_n - K, 0 \} 
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
$$
    

If you can, use Numba to speed up loops.

If possible, use Numba-based multithreading (`parallel=True`) to speed it even
further.




```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```


**Solution**


With $s_t := \ln S_t$, the price dynamics become

$$
s_{t+1} = s_t + \mu + \exp(h_t) \xi_{t+1}
$$

Using this fact, the solution can be written as follows.


```{code-cell} ipython3
from numpy.random import randn
M = 10_000_000

n, β, K = 20, 0.99, 100
μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0

@jit(parallel=True)
def compute_call_price_parallel(β=β,
                                μ=μ,
                                S0=S0,
                                h0=h0,
                                K=K,
                                n=n,
                                ρ=ρ,
                                ν=ν,
                                M=M):
    current_sum = 0.0
    # For each sample path
    for m in prange(M):
        s = np.log(S0)
        h = h0
        # Simulate forward in time
        for t in range(n):
            s = s + μ + np.exp(h) * randn()
            h = ρ * h + ν * randn()
        # And add the value max{S_n - K, 0} to current_sum
        current_sum += np.maximum(np.exp(s) - K, 0)
        
    return β**n * current_sum / M
```

Try swapping between `parallel=True` and `parallel=False` and noting the run time.

If you are on a machine with many CPUs, the difference should be significant.





