import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def small_besselj0(x):
    t = (x / 3.0)**2
    return (1.0 - 2.2499997*t + 1.2656208*(t**2) - 0.3163866*(t**3) + 
            0.0444479*(t**4) - 0.0039444*(t**5) + 0.0002100*(t**6))


def big_besselj0(x):
    p0 = 1 - 0.00109862*(8/x)**2 + 0.00002734*(8/x)**4
    q0 = -0.01562499*(8/x) + 0.00014304*(8/x)**3

    big_approx = jnp.sqrt(2/(jnp.pi * x)) * (p0 * jnp.cos(x - jnp.pi/4) - q0 * jnp.sin(x - jnp.pi/4))

    return big_approx


def small_besselj1(x):
    t = (x / 3.0)**2
    return x * (0.5 - 0.56249985*t + 0.21093573*(t**2) - 0.03954289*(t**3) + 
                0.00443319*(t**4) - 0.00031761*(t**5) + 0.00001109*(t**6))

def big_besselj1(x):
    f1 = 8/x

    p1 = 1 + 0.00183105*(f1)**2 - 0.00003516*(f1)**4 + 0.00000245*(f1**6)
    q1 = 0.04687499*(f1) - 0.00032337*(f1)**3 + 0.00001571*(f1**5)

    big_approx = jnp.sqrt(2/(jnp.pi * x)) * (p1 * jnp.cos(x - 3*jnp.pi/4) - q1 * jnp.sin(x - 3*jnp.pi/4))

    return big_approx

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
