import jax
import jax.numpy as jnp
import numpy as np
import time
from scipy.special import roots_legendre
from gauss_legendre import compute_nodes

# Ensure 64-bit precision
jax.config.update("jax_enable_x64", True)



# --- Runtime & Accuracy Test ---

def run_benchmark(n):
    print(f"\n--- Testing n = {n} ---")
    
    # 1. Warm-up / Compilation (First run for THIS specific n)
    start = time.perf_counter()
    _ = compute_nodes(n).block_until_ready()
    comp_time = time.perf_counter() - start
    print(f"JIT Compilation: {comp_time*1000:.2f} ms")

    # 2. Execution Time
    jax_times = []
    for _ in range(10): # More runs for better stats
        start = time.perf_counter()
        _ = compute_nodes(n).block_until_ready()
        jax_times.append(time.perf_counter() - start)
    avg_jax = np.mean(jax_times)
    
    # 3. SciPy baseline
    '''scipy_times = []    
    for _ in range(10): 
        start = time.perf_counter()
        scipy_nodes, _ = roots_legendre(n)
        scipy_times.append(time.perf_counter() - start)
    avg_scipy = np.mean(scipy_times)'''

    # 4. Accuracy
    jax_nodes = np.array(compute_nodes(n)) # Move to CPU for comparison
    # Ensure both are sorted the same way!
    #max_err = np.max(np.abs(np.sort(jax_nodes) - np.sort(scipy_nodes)))

    print(f"Avg JAX runtime:   {avg_jax*1000:.4f} ms")
    '''print(f"Avg SciPy runtime: {avg_scipy*1000:.4f} ms")
    print(f"Speedup:           {avg_scipy/avg_jax:.1f}x")
    print(f"Max Abs Error:     {max_err:.2e}")'''

#run_benchmark(500)
run_benchmark(1_000_000)
#run_benchmark(19879)
