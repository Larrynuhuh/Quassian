import time
import numpy as np
import jax
import jax.numpy as jnp
from gauss_legendre import integrate  

# 1. High-Frequency Integration Audit Block
def run_quassian_audit(n):
    # Overclocked Radar Chirp: Frequency sweeps from 0Hz to 1000Hz
    # Math: sin(2 * pi * (f_max / 2) * x^2) -> 2 * pi * 500.0 * x^2
    def extreme_radar_chirp(x):
        return jnp.sin(2.0 * jnp.pi * (500.0 * x**2))
    
    return integrate(extreme_radar_chirp, 0.0, 1.0, n)

# Wrap it in JIT
jit_audit = jax.jit(run_quassian_audit, static_argnums=0)

def test_quassian_extreme_chirp():
    print("=" * 60)
    print("     QUASSIAN OVERCLOCK CHECK: 1000Hz EXTRA-EXTREME CHIRP    ")
    print("=" * 60)

    a_limit = 0.0
    b_limit = 1.0
    n_nodes = 3000  # Keeping it at 1,000 nodes to see if precision holds

    # Hyper-accurate baseline verified via analytical Fresnel Integrals
    # For int_0^1 sin(2*pi*500*x^2) dx
    true_integral = 0.011021184956501

    print(f"Test Setup:")
    print(f"-> Function: f(x) = sin(2 * pi * 500 * x^2)  [0Hz -> 1000Hz]")
    print(f"-> Domain:   [{a_limit}, {b_limit}]")
    print(f"-> Nodes (n): {n_nodes}\n")

    # 1. Warm-up and compilation profile
    print("Status: Warming up JAX JIT compiler for new frequency graph...")
    try:
        start_compile = time.perf_counter()
        _ = jit_audit(n_nodes).block_until_ready()
        compile_time = time.perf_counter() - start_compile
        print(f"Status: Compilation complete in {compile_time:.4f} seconds.\n")
    except Exception as e:
        print(f"⚠️ Compilation Error: {e}\n")
        return

    # 2. Pure Hardware Latency Measurement
    print("Status: Measuring hardware response under intense compression...")
    try:
        start_latency = time.perf_counter()
        quassian_ans_device = jit_audit(n_nodes).block_until_ready()
        hardware_latency = time.perf_counter() - start_latency
        
        quassian_ans = float(quassian_ans_device)
        abs_error = abs(quassian_ans - true_integral)
        
        print(f"\n-> Verified Master Baseline : {true_integral:.15f}")
        print(f"-> Quassian Engine Output   : {quassian_ans:.15f}")
        print(f"-> Absolute Precision Error : {abs_error:.2e}")
        print(f"-> Pure Hardware Latency    : {hardware_latency * 1000:.3f} ms")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return
    print("=" * 60)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    test_quassian_extreme_chirp()
