import time
import jax
import jax.numpy as jnp
from jax import jit
from gauss_legendre import integrate

# =====================================================================
# 0. PLACEHOLDER FOR YOUR LIBRARY FUNCTION
# =====================================================================


# =====================================================================
# 1. MATHEMATICAL TEST CASES (Testing Code Integrity)
# =====================================================================

# Test Case A: A highly oscillatory physics wave (Tests granularity/aliasing)
# Analytical integral of cos(x) from 0 to pi = 0
def oscillatory_func(x):
    return jnp.cos(10 * x)
true_oscillatory = jnp.sin(10 * jnp.pi) / 10  # 0.0

# Test Case B: Singularity / Discontinuity handling
# Integral of 1 / sqrt(x) from 0.01 to 1 = 1.8
def singularity_func(x):
    return 1.0 / jnp.sqrt(x)
true_singularity = 2.0 * (jnp.sqrt(1.0) - jnp.sqrt(0.01))

# =====================================================================
# 2. MULTI-DIMENSIONAL EXTENSION (Lines, Surfaces, Volumes)
# =====================================================================

# Dimensional Test 1: Line Integral (Arc Length of a Helix)
# Curve: r(t) = (cos(t), sin(t), t) -> |r'(t)| = sqrt(sin^2 + cos^2 + 1) = sqrt(2)
# Integral of sqrt(2) from 0 to 2*pi
def helix_line_element(t):
    return jnp.sqrt(jnp.sin(t)**2 + jnp.cos(t)**2 + 1.0)
true_line = jnp.sqrt(2.0) * 2.0 * jnp.pi

# Dimensional Test 2: Surface Integral (Area of a flat region / Curved Surface)
# We nest the 1D integrator to perform an iterated 2D integral over [0,1]x[0,1]
# Function: f(x,y) = x * y. True Answer = 0.25
def surface_integral_2d(n_points):
    def inner_y_integral(x):
        # For a fixed x, integrate over y from 0 to 1
        return integrate(lambda y: x * y, 0.0, 1.0, n_points)
    
    # Vectorize the inner step so the outer 1D integrator can process it as an array
    vectorized_inner = jax.vmap(inner_y_integral)
    return integrate(vectorized_inner, 0.0, 1.0, n_points)
true_surface = 0.25

# Dimensional Test 3: Volume Integral (Mass of a density cube)
# Nested iterated 3D integral over [0,1]x[0,1]x[0,1]
# Density function: rho(x,y,z) = x * y * z. True Answer = 1/8 = 0.125
def volume_integral_3d(n_points):
    def inner_z(x, y):
        return integrate(lambda z: x * y * z, 0.0, 1.0, n_points)
    
    def inner_y(x):
        # vmap over y for a fixed x
        return integrate(lambda y: jax.vmap(inner_z, in_axes=(None, 0))(x, y), 0.0, 1.0, n_points)
        
    vectorized_outer = jax.vmap(inner_y)
    return integrate(vectorized_outer, 0.0, 1.0, n_points)
true_volume = 0.125


# =====================================================================
# 3. ACCURACY EVALUATION ENGINE
# =====================================================================
print("--- ACCURACY & CAPABILITY REPORT ---")
N_TEST = 2000

res_osc = integrate(oscillatory_func, 0.0, jnp.pi, N_TEST)
print(f"Line (Oscillatory)  | Calculated: {res_osc:.6f} | True: {true_oscillatory:.6f} | Abs Error: {abs(res_osc - true_oscillatory):.2e}")

res_sing = integrate(singularity_func, 0.01, 1.0, N_TEST)
print(f"Line (Singularity)  | Calculated: {res_sing:.6f} | True: {true_singularity:.6f} | Abs Error: {abs(res_sing - true_singularity):.2e}")

res_helix = integrate(helix_line_element, 0.0, 2.0*jnp.pi, N_TEST)
print(f"Line (Helix Length) | Calculated: {res_helix:.6f} | True: {true_line:.6f} | Abs Error: {abs(res_helix - true_line):.2e}")

# Surface and volume tests require fewer points per dimension to avoid memory explosions (O(N^2) and O(N^3))
res_surf = surface_integral_2d(200)
print(f"Surface (2D Area)   | Calculated: {res_surf:.6f} | True: {true_surface:.6f} | Abs Error: {abs(res_surf - true_surface):.2e}")

res_vol = volume_integral_3d(100)
print(f"Volume (3D Mass)    | Calculated: {res_vol:.6f} | True: {true_volume:.6f} | Abs Error: {abs(res_vol - true_volume):.2e}")


# =====================================================================
# 4. HPC SPEED BENCHMARKING ENGINE
# =====================================================================
print("\n--- PERFORMANCE & BENCHMARK REPORT ---")

# We wrap the target test inside a JIT compilation to evaluate pure hardware performance
@jit(static_argnums=(0))
def benchmark_target(n):
    return integrate(oscillatory_func, 0.0, jnp.pi, n)

# Array sizes to stress-test memory pipelines
problem_sizes = [10_0, 100_0, 1_000_0]

for size in problem_sizes:
    # Warm-up / Compilation Run (Crucial for JAX benchmarking)
    start_compile = time.perf_counter()
    _ = benchmark_target(size).block_until_ready() # block_until_ready forces asynchronous JAX execution to finish
    compile_time = time.perf_counter() - start_compile
    
    # Pure Execution Run (Hot Run)
    start_hot = time.perf_counter()
    result = benchmark_target(size).block_until_ready()
    hot_time = time.perf_counter() - start_hot
    
    print(f"Elements: {size:12,} | Compile Time: {compile_time:.4f}s | Hot Execution: {hot_time:.6f}s")
