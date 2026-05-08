import jax
import jax.numpy as jnp
import numpy as np
import time
from scipy.special import roots_legendre

# Ensure 64-bit precision
jax.config.update("jax_enable_x64", True)

# --- Your Implementation ---

def j0smallroots():
    return jnp.array([2.4048255576957728, 5.5200781102863106, 8.6537279129110122, 11.791534439014281, 14.930917708487787,
    18.071063967910922, 21.211636629879258, 24.352471530749302, 27.493479132040254, 30.634606468431975], dtype=jnp.float64)

def macmahon(n):
    k = jnp.arange(11, n + 1)
    beta = (k - 0.25) * jnp.pi
    return beta + (1/(8*beta)) - (31/(384*beta**3)) + (3779/(15360*beta**5))


@jax.jit(static_argnames=['n'])
def compute_nodes(n):
    num_half = n // 2
    k = jnp.arange(1, num_half + 1)
    
    # Calculate regimes
    interior = interior_nodes(n)
    boundary = boundary_nodes(n)
    
    # Handoff condition
    cond = k <= (n**(2/3) / jnp.pi)
    pos_nodes = jnp.where(cond, boundary, interior)
    
    # Symmetrize
    neg_nodes = -pos_nodes[::-1]
    if n % 2 == 0:
        return jnp.concatenate([neg_nodes, pos_nodes])
    else:
        return jnp.concatenate([neg_nodes, jnp.array([0.0]), pos_nodes])

# --- Runtime & Accuracy Test ---

def run_benchmark(n):
    print(f"\n--- Testing n = {n} ---")
    
    # 1. Warm-up / Compilation
    start = time.perf_counter()
    _ = compute_nodes(n).block_until_ready()
    comp_time = time.perf_counter() - start
    print(f"JIT Compilation: {comp_time*1000:.2f} ms")

    # 2. Execution Time (Average of 10 runs)
    jax_times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = compute_nodes(n).block_until_ready()
        jax_times.append(time.perf_counter() - start)
    avg_jax = np.mean(jax_times)
    
    # 3. SciPy baseline
    start = time.perf_counter()
    scipy_nodes, _ = roots_legendre(n)
    scipy_time = time.perf_counter() - start

    # 4. Accuracy Check
    jax_nodes = compute_nodes(n)
    max_err = np.max(np.abs(jax_nodes - scipy_nodes))

    print(f"Avg JAX runtime:   {avg_jax*1000:.4f} ms")
    print(f"SciPy runtime:     {scipy_time*1000:.4f} ms")
    print(f"Speedup:           {scipy_time/avg_jax:.1f}x")
    print(f"Max Abs Error:     {max_err:.2e}")

# Run tests
run_benchmark(100)
run_benchmark(1000)
run_benchmark(10000)
