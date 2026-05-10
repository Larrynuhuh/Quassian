import jax
import jax.numpy as jnp
import scipy
import numpy as np
jax.config.update("jax_enable_x64", True)

def scipy_j0(x):
    x = np.asarray(x, dtype=np.float64)
    return scipy.special.j0(x)

def bessel_j0(x):
    out_shape = jax.ShapeDtypeStruct(x.shape, jnp.float64)
    return jax.pure_callback(scipy_j0, out_shape, x)

def scipy_j1(x):
    x = np.asarray(x, dtype=np.float64)
    return scipy.special.j1(x)

def bessel_j1(x):
    out_shape = jax.ShapeDtypeStruct(x.shape, jnp.float64)
    return jax.pure_callback(scipy_j1, out_shape, x)

def macmahon(n):
    if n % 2 == 0: 
        m = n // 2
    else:
        m = n // 2 + 1
    k = jnp.arange(11, m+1)
    beta = (k - 0.25) * jnp.pi
    expression = beta + (1/(8*beta)) - (31/(384*beta**3)) + (3779/(15360*beta**5))

    return expression

def j0smallroots():
    return jnp.array([2.4048255576957728, 5.5200781102863106, 8.6537279129110122, 11.791534439014281, 14.930917708487787,
    18.071063967910922, 21.211636629879258, 24.352471530749302, 27.493479132040254, 30.634606468431975], dtype=jnp.float64)

def interior_nodes(n):
    if n % 2 == 0: 
        m = n // 2
    else:
        m = n // 2 + 1
    k = jnp.arange(1, m+1)
    phi_k = ((k-0.25) * jnp.pi) / (n + 0.5)

    interior = (1 - ((n-1)/(8*n**3)) - 1/(384*n**4) * (39 - (28/(jnp.sin(phi_k)**2)))) * jnp.cos(phi_k)

    return interior

def boundary_nodes(n):
    if n % 2 == 0: 
        m = n // 2
    else:
        m = n // 2 + 1
    k = jnp.arange(1, m+1)

    j_k = jnp.concatenate((j0smallroots(), macmahon(n)))
    j_k = j_k[:m]

    psi_k = j_k/(n + 0.5)

    boundary_term = psi_k + ((psi_k * 1/jnp.tan(psi_k)) - 1) / (8 * psi_k * (n + 0.5)**2) 
    boundary = jnp.cos(boundary_term)

    return boundary

def compute_nodes(n):
    if n % 2 == 0: 
        m = n // 2
    else:
        m = n // 2 + 1
    k = jnp.arange(1, m+1)

    interior = interior_nodes(n)
    boundary = boundary_nodes(n)

    cond = (m - k + 1) <= jnp.ceil(m**(2/3)/jnp.pi).astype(int)

    nodes_raw = jnp.where(cond, boundary, interior)
    nodes_raw = nodes_raw[::-1]
    
    if n % 2 == 0:
        nodes = jnp.concatenate((-nodes_raw[::-1], nodes_raw))
    else:
        nodes = jnp.concatenate((-nodes_raw[:0:-1], nodes_raw))

    return nodes  


def interior_asymptotic(n, x):
    log_ratio = jnp.exp(jax.scipy.special.gammaln(n + 1.0) - jax.scipy.special.gammaln(n + 1.5))
    Cn = jnp.sqrt(4.0 / jnp.pi) * log_ratio
    
    m = jnp.array([0.0, 1.0, 2.0])[:, None]
    theta = jnp.arccos(x)[None, :]
    hnm = jnp.array([1, 0.25/(n + 1.5), 9/(32.0 * (n + 1.5) * (n + 2.5))])[:, None]
    anm = theta * (n + m + 0.5) - (m + 1/2) * jnp.pi/2

    expression = jnp.sum(Cn * hnm * (jnp.cos(anm)/(2 * jnp.sin(theta))**(m + 0.5)), axis=0)
    return expression

def boundary_asymptotic(n, x): 
    theta = jnp.arccos(x)
    k = jnp.arange(1, x.shape[0]+1)

    rho = n + 0.5

    def g(x): return (x * (1/jnp.tan(x)) - 1) / (2 * x)
    A0 = 1.0
    A1 = (1/8 * jax.vmap(jax.grad(g))(theta)) - (1/8 * g(theta)/theta) - (1/32 * g(theta)**2)
    B0 = 1/4 * g(theta)

    j0 = bessel_j0(rho * theta)
    j1 = bessel_j1(rho * theta)

    term1 = jnp.sqrt(theta/jnp.sin(theta)) 
    term2 = j0 * (A0 + (A1/rho**2)) + (theta * j1 * (B0/rho))

    return term1 * term2


def poly_eval(n, x):
    bound = boundary_asymptotic(n, x)
    interior = interior_asymptotic(n, x)
    theta = jnp.arccos(x)

    cond = (jnp.pi/6 >= theta) | (theta >= 5*jnp.pi/6)
    return jnp.where(cond, bound, interior)

def f(theta, x): return poly_eval(theta, x)
    
def derivative_eval(n, x):
    Pn = poly_eval(n, x)
    Pn_1 = poly_eval(n-1, x)

    theta = jnp.arccos(x)
    term = ((n * jnp.cos(theta) * Pn) - (n * Pn_1))/jnp.sin(theta)

    return term


def newtoned_nodes(n):
    nodes = compute_nodes(n)
    theta = jnp.arccos(nodes)

    pn_der = -derivative_eval(n, jnp.cos(theta))/jnp.sin(theta)
    nodes_1 = nodes - poly_eval(n, jnp.cos(theta))/pn_der
    pn_der1 = -derivative_eval(n, nodes_1)/jnp.sin(theta)
    nodes_2 = nodes_1 - poly_eval(n, nodes_1)/pn_der1

    return nodes_2

def compute_weights(n, x):
    return 2/(derivative_eval(n, x)**2)

@jax.jit(static_argnums=(0, 3))
def integrate(func, a, b, n):
    nodes = newtoned_nodes(n)
    weights = compute_weights(n, nodes)
    bounded_nodes = ((b+a)/2 + ((b-a)/2) * nodes)

    integral = ((b-a)/2) * jnp.sum(func(bounded_nodes) * weights, axis=0)
    return integral


# The Runge "Polynomial Killer"
test_func = lambda x: 1 / (1 + 2500 * x**2)
a, b = -1, 1

# True value: (2/sqrt(2500)) * arctan(sqrt(2500))
# For 2500, that is (2/50) * arctan(50)
true_value = 0.062035459261058

n_test = 1000 # Your method should handle this easily
result = integrate(test_func, a, b, n_test)

print(f"Testing Steep Runge (n = {n_test})")
print(f"Calculated: {result:.16f}")
print(f"Actual:     {true_value:.16f}")
print(f"Error:      {abs(result - true_value):.2e}")
