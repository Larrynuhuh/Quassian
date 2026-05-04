import jax
import jax.numpy as jnp

@jax.jit(static_argnums=(0, 3))
def integrate(func, a, b, n):
    nodes, weights = jnp.polynomial.legendre.leggauss(n)
    bounded_nodes = ((b+a)/2 + ((b-a)/2) * nodes)
    xs = bounded_nodes * weights

    def body_fun(state, x):
        new_state = state + x
        return new_state, new_state
    
    integral_raw, integral_traj = jax.lax.scan(body_fun, 0, xs)

    integral = (b-a)/2 * integral_raw
    return integral 

func = lambda x: x**5 - 2*x**4 + 3*x**3 - 4*x**2 + 5*x - 6
a, b = 0, 5
n = 4

print(f"Integral of func from {a} to {b} with n={n}: {integrate(func, a, b, n)}")

