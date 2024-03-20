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

# Fun with JAX

#### [John Stachurski](https://johnstachurski.net/) and [Chase Coleman](https://github.com/cc7768)
March 2024

This notebook illustrates the power of [JAX](https://github.com/google/jax), a Python library built by Google Research.

It should be run on a machine with a GPU --- for example, try Google Colab with the runtime environment set to include a GPU.

The aim is just to give a small taste of high performance computing in Python -- details will be covered later in the course.

+++

We start with some imports

```{code-cell} ipython3
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
```

Let's check our hardware:

```{code-cell} ipython3
!nvidia-smi
```

```{code-cell} ipython3
!lscpu -e
```

## Transforming Data

+++

A very common numerical task is to apply a transformation to a set of data points.

Our transformation will be the cosine function.

+++

Here we evaluate the cosine function at 50 points.

```{code-cell} ipython3
x = np.linspace(0, 10, 50)
y = np.cos(x)
```

Let's plot.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()
```

Our aim is to evaluate the cosine function at many points.

```{code-cell} ipython3
n = 50_000_000
x = np.linspace(0, 10, n)
```

### With NumPy

```{code-cell} ipython3
%time np.cos(x)
```

```{code-cell} ipython3
%time np.cos(x)
```

```{code-cell} ipython3
x = None   # delete x
```

### With JAX

```{code-cell} ipython3
x_jax = jnp.linspace(0, 10, n)
```

```{code-cell} ipython3
%time jnp.cos(x_jax).block_until_ready()
```

```{code-cell} ipython3
%time jnp.cos(x_jax).block_until_ready()
```

Can you explain why the timing changes after we change sizes?

```{code-cell} ipython3
x_jax = jnp.linspace(0, 10, n + 1)
```

```{code-cell} ipython3
%time jnp.cos(x_jax).block_until_ready()
```

```{code-cell} ipython3
%time jnp.cos(x_jax).block_until_ready()
```

```{code-cell} ipython3
x_jax = None
```

## Evaluating a more complicated function

```{code-cell} ipython3
def f(x):
    y = np.cos(2 * x**2) + np.sqrt(np.abs(x)) + 2 * np.sin(x**4) - 0.1 * x**2
    return y
```

```{code-cell} ipython3
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, f(x))
ax.scatter(x, f(x))
plt.show()
```

Now let's try with a large array.

+++

### With NumPy

```{code-cell} ipython3
x = np.linspace(0, 10, n)
```

```{code-cell} ipython3
%time f(x)
```

```{code-cell} ipython3
%time f(x)
```

### With JAX

```{code-cell} ipython3
def f(x):
    y = jnp.cos(2 * x**2) + jnp.sqrt(jnp.abs(x)) + 2 * jnp.sin(x**4) - x**2
    return y
```

```{code-cell} ipython3
x_jax = jnp.linspace(0, 10, n)
```

```{code-cell} ipython3
%time f(x_jax).block_until_ready()
```

```{code-cell} ipython3
%time f(x_jax).block_until_ready()
```

### Compiling the Whole Function

```{code-cell} ipython3
f_jax = jax.jit(f)
```

```{code-cell} ipython3
%time f_jax(x_jax).block_until_ready()
```

```{code-cell} ipython3
%time f_jax(x_jax).block_until_ready()
```

## Solving Linear Systems

```{code-cell} ipython3
np.random.seed(1234)
n = 5_000
A = np.random.randn(n, n)
b = np.ones(n)
```

```{code-cell} ipython3
%time np.linalg.solve(A, b)
```

```{code-cell} ipython3
A, b = [jax.device_put(v) for v in (A, b)]
```

```{code-cell} ipython3
%time jnp.linalg.solve(A, b).block_until_ready()
```

```{code-cell} ipython3
%time jnp.linalg.solve(A, b).block_until_ready()
```

## Nonlinear Equations

+++

In many cases we want to solve a system of nonlinear equations.

This section gives an example --- solving for an equilibrium price vector when supply and demand are nonlinear.

We start with a simple two good market.

Then we shift up to high dimensions.

We will see that, in high dimensions, automatic differentiation and the GPU are very helpful.

+++

### A Two Goods Market Equilibrium

Let’s start by computing the market equilibrium of a two-good problem.

Our first step is to define the excess demand function

$$
e(p) = 
    \begin{pmatrix}
    e_0(p) \\
    e_1(p)
    \end{pmatrix}
$$

The function below calculates the excess demand for given parameters

```{code-cell} ipython3
:hide-output: false

def e(p, A, b, c):
    "Excess demand is demand - supply at price vector p"
    return np.exp(- A @ p) + c - b * np.sqrt(p)
```

Our default parameter values will be

$$
A = \begin{pmatrix}
            0.5 & 0.4 \\
            0.8 & 0.2
        \end{pmatrix},
            \qquad 
    b = \begin{pmatrix}
            1 \\
            1
        \end{pmatrix}
    \qquad \text{and} \qquad
    c = \begin{pmatrix}
            1 \\
            1
        \end{pmatrix}
$$

```{code-cell} ipython3
:hide-output: false

A = np.array(((0.5, 0.4),
              (0.8, 0.2)))
b = np.ones(2)
c = np.ones(2)
```

Next we plot the two functions $ e_0 $ and $ e_1 $ on a grid of $ (p_0, p_1) $ values, using contour surfaces and lines.

We will use the following function to build the contour plots

```{code-cell} ipython3
:hide-output: false

def plot_excess_demand(ax, good=0, grid_size=100, grid_max=4, surface=True):
    p_grid = np.linspace(0, grid_max, grid_size)
    z = np.empty((100, 100))

    for i, p_1 in enumerate(p_grid):
        for j, p_2 in enumerate(p_grid):
            z[i, j] = e((p_1, p_2), A, b, c)[good]

    if surface:
        cs1 = ax.contourf(p_grid, p_grid, z.T, alpha=0.5)
        plt.colorbar(cs1, ax=ax, format="%.6f")

    ctr1 = ax.contour(p_grid, p_grid, z.T, levels=[0.0])
    ax.set_xlabel("$p_0$")
    ax.set_ylabel("$p_1$")
    ax.set_title(f'Excess Demand for Good {good}')
    plt.clabel(ctr1, inline=1, fontsize=13)
```

Here’s our plot of $ e_0 $:

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
plot_excess_demand(ax, good=0)
plt.show()
```

Here’s our plot of $ e_1 $:

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
plot_excess_demand(ax, good=1)
plt.show()
```

We see the black contour line of zero, which tells us when $ e_i(p)=0 $.

For a price vector $ p $ such that $ e_i(p)=0 $ we know that good $ i $ is in equilibrium (demand equals supply).

If these two contour lines cross at some price vector $ p^* $, then $ p^* $ is an equilibrium price vector.

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
for good in (0, 1):
    plot_excess_demand(ax, good=good, surface=False)
plt.show()
```

It seems there is an equilibrium close to $ p = (1.6, 1.5) $.

+++

#### Using a Multidimensional Root Finder

To solve for $ p^* $ more precisely, we use a zero-finding algorithm from `scipy.optimize`.

We supply $ p = (1, 1) $ as our initial guess.

```{code-cell} ipython3
:hide-output: false

init_p = np.ones(2)
```

This uses the [modified Powell method](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html#optimize-root-hybr) to find the zero

```{code-cell} ipython3
:hide-output: false

%%time
solution = scipy.optimize.root(lambda p: e(p, A, b, c), init_p, method='hybr')
```

Here’s the resulting value:

```{code-cell} ipython3
:hide-output: false

p = solution.x
p
```

This looks close to our guess from observing the figure. We can plug it back into $ e $ to test that $ e(p) \approx 0 $:

```{code-cell} ipython3
:hide-output: false

np.max(np.abs(e(p, A, b, c)))
```

This is indeed a very small error.

+++

#### Adding Gradient Information

In many cases, for zero-finding algorithms applied to smooth functions, supplying the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of the function leads to better convergence properties.

Here we manually calculate the elements of the Jacobian

$$
J(p) = 
    \begin{pmatrix}
        \frac{\partial e_0}{\partial p_0}(p) & \frac{\partial e_0}{\partial p_1}(p) \\
        \frac{\partial e_1}{\partial p_0}(p) & \frac{\partial e_1}{\partial p_1}(p)
    \end{pmatrix}
$$

```{code-cell} ipython3
:hide-output: false

def jacobian_e(p, A, b, c):
    p_0, p_1 = p
    a_00, a_01 = A[0, :]
    a_10, a_11 = A[1, :]
    j_00 = -a_00 * np.exp(-a_00 * p_0) - (b[0]/2) * p_0**(-1/2)
    j_01 = -a_01 * np.exp(-a_01 * p_1)
    j_10 = -a_10 * np.exp(-a_10 * p_0)
    j_11 = -a_11 * np.exp(-a_11 * p_1) - (b[1]/2) * p_1**(-1/2)
    J = [[j_00, j_01],
         [j_10, j_11]]
    return np.array(J)
```

```{code-cell} ipython3
:hide-output: false

solution = scipy.optimize.root(lambda p: e(p, A, b, c),
                init_p, 
                jac=lambda p: jacobian_e(p, A, b, c), 
                method='hybr')
```

Now the solution is even more accurate (although, in this low-dimensional problem, the difference is quite small):

```{code-cell} ipython3
:hide-output: false

p = solution.x
np.max(np.abs(e(p, A, b, c)))
```

#### Newton’s Method via JAX

We use a multivariate version of Newton’s method to compute the equilibrium price.

The rule for updating a guess $ p_n $ of the equilibrium price vector is


<a id='equation-multi-newton'></a>
$$
p_{n+1} = p_n - J_e(p_n)^{-1} e(p_n) \tag{3.1}
$$

Here $ J_e(p_n) $ is the Jacobian of $ e $ evaluated at $ p_n $.

Iteration starts from initial guess $ p_0 $.

Instead of coding the Jacobian by hand, we use automatic differentiation via `jax.jacobian()`.

```{code-cell} ipython3
:hide-output: false

def newton(f, x_0, tol=1e-5, max_iter=15):
    """
    A multivariate Newton root-finding routine.

    """
    x = x_0
    f_jac = jax.jacobian(f)
    @jax.jit
    def q(x):
        " Updates the current guess. "
        return x - jnp.linalg.solve(f_jac(x), f(x))
    error = tol + 1
    n = 0
    while error > tol:
        n += 1
        if(n > max_iter):
            raise Exception('Max iteration reached without convergence')
        y = q(x)
        error = jnp.linalg.norm(x - y)
        x = y
        print(f'iteration {n}, error = {error}')
    return x
```

```{code-cell} ipython3
:hide-output: false

def e(p, A, b, c):
    return jnp.exp(- A @ p) + c - b * jnp.sqrt(p)
```

```{code-cell} ipython3
:hide-output: false

p = newton(lambda p: e(p, A, b, c), init_p)
p
```

```{code-cell} ipython3
:hide-output: false

jnp.max(jnp.abs(e(p, A, b, c)))
```

### A High-Dimensional Problem

Let’s now apply the method just described to investigate a large market with 5,000 goods.

We randomly generate the matrix $ A $ and set the parameter vectors $ b, c $ to $ 1 $.

```{code-cell} ipython3
:hide-output: false

dim = 5_000
seed = 32

# Create a random matrix A and normalize the rows to sum to one
key = jax.random.PRNGKey(seed)
A = jax.random.uniform(key, (dim, dim))
s = jnp.sum(A, axis=0)
A = A / s

# Set up b and c
b = jnp.ones(dim)
c = jnp.ones(dim)
```

Here’s our initial condition $ p_0 $

```{code-cell} ipython3
:hide-output: false

init_p = jnp.ones(dim)
```

By combining the power of Newton’s method, JAX accelerated linear algebra,
automatic differentiation, and a GPU, we obtain a relatively small error for
this high-dimensional problem in just a few seconds:

```{code-cell} ipython3
:hide-output: false

%time p = newton(lambda p: e(p, A, b, c), init_p).block_until_ready()
```

Here’s the size of the error:

```{code-cell} ipython3
:hide-output: false

jnp.max(jnp.abs(e(p, A, b, c)))
```

With the same tolerance, SciPy’s `root` function takes much longer to run.

```{code-cell} ipython3
:hide-output: false

%%time

solution = scipy.optimize.root(lambda p: e(p, A, b, c),
                init_p,
                method='hybr',
                tol=1e-5)
```

```{code-cell} ipython3
:hide-output: false

p = solution.x
jnp.max(jnp.abs(e(p, A, b, c)))
```

```{code-cell} ipython3

```
