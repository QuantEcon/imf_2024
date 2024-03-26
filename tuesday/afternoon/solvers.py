import jax
import jax.numpy as jnp

# Use 64 bit floats with JAX in order to increase precision.

jax.config.update("jax_enable_x64", True)


default_tolerance = 1e-9
default_max_iter = int(1e6)

def successive_approx(f,
                      x_init,
                      tol=default_tolerance,
                      max_iter=default_max_iter,
                      verbose=True,
                      print_skip=1000):

    "Uses successive approximation on f."

    if verbose:
        print("Beginning iteration\n\n")

    current_iter = 0
    x = x_init
    error = tol + 1
    while error > tol and current_iter < max_iter:
        x_new = f(x)
        error = jnp.max(jnp.abs(x_new - x))
        if verbose and current_iter % print_skip == 0:
            print("iter = {}, error = {}".format(current_iter, error))
        current_iter += 1
        x = x_new

    if current_iter == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        if verbose:
            print(f"Iteration converged after {current_iter} iterations")

    return x


def newton_solver(f, 
                  x_init, 
                  tol=default_tolerance, 
                  max_iter=default_max_iter,
                  bicgstab_atol=1e-6,
                  verbose=True,
                  print_skip=1):
    """
    Apply Newton's algorithm to find a fixed point of f. The routine defines 
    g via g(x) = f(x) - x and then searches for a root of g via Newton's
    method, which iterates on 

        x_{n+1} = x_n - J(x_n)^{-1} g(x_n)

    until convergence, where J(x) is the Jacobian of g at x. The implementation 
    below defines 

        q(x) := x - J(x)^{-1} g(x)

    and passes this function to successive_approx.

    To compute J(x)^{-1} g(x) we can in principle use
    `jnp.linalg.solve(jax.jacobian(g)(x), g(x))`. However, this operation is
    very memory intensive when x is high-dimensional. It also requires that g
    is a regular 2D array (matrix), which necessitates conversion to a single
    index. 

    To avoid instantiating the large matrix J(x), we use jax.jvp to define the
    linear map v -> J(x) v. This map is computed on demand for any given v,
    which avoids instantiating J(x).  We then pass this to a solver that can
    invert arbitrary linear maps.
    """
    g = lambda x: f(x) - x
    @jax.jit
    def q(x):
        # First we define the map v -> J(x) v from x and g
        jac_x_prod = lambda v: jax.jvp(g, (x,), (v,))[1]
        # Next we compute J(x)^{-1} g(x).  Currently we use 
        # sparse.linalg.bicgstab. Another option is sparse.linalg.bc
        # but this operation seems to be less stable.
        b = jax.scipy.sparse.linalg.bicgstab(
                jac_x_prod, g(x), 
                atol=bicgstab_atol)[0]
        return x - b
    return successive_approx(q, x_init, tol, max_iter, verbose, print_skip)

