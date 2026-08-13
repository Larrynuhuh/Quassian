import time
import numpy as np
from gauss_legendre import integrate
import jax
import jax.numpy as jnp

# Define the function outside JIT first
def run_quassian_audit(n):
    # Testing with your new high-frequency chirp function
    def brutal_chirp(x):
        return jnp.sin(500.0 / (x + 0.1))
    return integrate(brutal_chirp, 0.0, 1.0, n)

# Wrap it in JIT, explicitly telling JAX that argument 0 (n) is a static scalar integer
jit_audit = jax.jit(run_quassian_audit, static_argnums=0)

print("=" * 60)
print("FIXED COMPILER AUDIT: QUASSIAN QUADRATURE ENGINE")
print("=" * 60)

try:
    # Now lowering with a static integer 10000 works flawlessly
    lowered = jit_audit.lower(4000)
    hlo_text = lowered.as_text()
    
    print("\n--- XLA HLO TEXT PREVIEW (First 40 lines) ---")
    lines = hlo_text.splitlines()
    for line in lines[:40]:
        print(f"  {line}")
    if len(lines) > 40:
        print(f"  ... [{len(lines)-40} compiler lines hidden] ...")
        
    # Extract Memory Traffic and Hardware Operational Costs
    compiled_module = lowered.compile()
    analysis = compiled_module.cost_analysis()
    
    print("\n--- HARDWARE & MEMORY ANALYSIS ---")
    print(f"  Total FLOPS     : {analysis.get('flops', 'N/A')}")
    print(f"  Memory Traffic  : {analysis.get('bytes accessed', 'N/A')} bytes")
    
except Exception as e:
    print(f"❌ Audit failed: {e}")

import time
import numpy as np
from gauss_legendre import integrate
import jax
import jax.numpy as jnp

def test_quassian_brutal_chirp():
    print("=" * 60)
    print("      QUASSIAN ACCURACY CHECK: BRUTAL HIGH-FREQ CHIRP     ")
    print("=" * 60)

    # 1. The Highly Oscillatory Stress-Test Function
    # Frequency accelerates massively as x hits 0
    def brutal_chirp(x):
        return jnp.sin(500.0 / (x + 0.1))
    
    a_limit = 0.0
    b_limit = 1.0
    
    # Highly precise analytical baseline value (verified via 30-dps mpmath)
    true_integral = -0.00133113185029490

    # Choose your node resolution (e.g., 10000, 15000, or 20000)
    n_nodes = 4000

    print(f"Test Setup:")
    print(f"-> Function: f(x) = sin(500 / (x + 0.1))")
    print(f"-> Domain:   [{a_limit}, {b_limit}]")
    print(f"-> Nodes (n): {n_nodes}\n")

    # 2. Warm up the JAX JIT compiler
    print("Status: Warming up JAX JIT compiler for Quassian...")
    try:
        # Wrap it with static_argnums to avoid the Tracer/Hash error
        jit_integrate = jax.jit(integrate, static_argnums=(0,3))
        _ = jit_integrate(brutal_chirp, a_limit, b_limit, n_nodes)
        print("Status: Compilation complete.\n")
    except Exception as e:
        print(f"⚠️ Compilation Error: {e}\n")
        return

    # 3. Accuracy Evaluation
    print("Status: Evaluating absolute precision limits...")
    try:
        quassian_ans = float(jit_integrate(brutal_chirp, a_limit, b_limit, n_nodes))
        abs_error = abs(quassian_ans - true_integral)
        
        print(f"\n-> Verified Master Baseline : {true_integral:.15f}")
        print(f"-> Quassian Engine Output   : {quassian_ans:.15f}")
        print(f"-> Absolute Precision Error : {abs_error:.2e}")
    except Exception as e:
        print(f"❌ Execution failed during evaluation: {e}")
        return

    print("=" * 60)

if __name__ == "__main__":
    # Ensure float64 is globally active
    jax.config.update("jax_enable_x64", True)
    test_quassian_brutal_chirp()

import time

# 1. Warm-up Run (Measures compilation + first execution)
print("Status: Compiling and warming up XLA pipeline...")
start_compile = time.perf_counter()

# Trigger your engine and block until the hardware finishes
_ = jit_audit(4000).block_until_ready() 

compile_time = time.perf_counter() - start_compile
print(f"-> Compilation + Warmup Time: {compile_time:.4f} seconds")

# 2. Pure Latency Run (Measures raw hardware execution speed)
print("\nStatus: Measuring pure hardware latency...")
start_latency = time.perf_counter()

# This runs entirely on the compiled XLA binary
result = jit_audit(4000).block_until_ready()

hardware_latency = time.perf_counter() - start_latency
print(f"-> Pure Engine Hardware Latency: {hardware_latency * 1000:.3f} ms")
