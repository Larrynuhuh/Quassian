import time
import numpy as np
import jax
import jax.numpy as jnp
from gauss_legendre import integrate  # Your core global integrator

# 1. The Global Integration Audit Block
def run_quassian_audit(n):
    # The Linear Radar Chirp: Frequency sweeps from 0Hz to 250Hz
    def radar_chirp(x):
        return jnp.sin(2.0 * jnp.pi * (125.0 * x**2))
    
    return integrate(radar_chirp, 0.0, 1.0, n)

# Wrap it in JIT
jit_audit = jax.jit(run_quassian_audit, static_argnums=0)

def test_quassian_radar_chirp():
    print("=" * 60)
    print("      QUASSIAN ACCURACY CHECK: REAL DSP RADAR CHIRP       ")
    print("=" * 60)

    a_limit = 0.0
    b_limit = 1.0
    n_nodes = 15000  # Throwing your elite 15k node budget at it

    # Hyper-accurate baseline verified via analytical Fresnel Integrals
    # For int_0^1 sin(2*pi*125*x^2) dx
    true_integral = 0.02172406077665616

    print(f"Test Setup:")
    print(f"-> Function: f(x) = sin(2 * pi * 125 * x^2)  [0Hz -> 250Hz]")
    print(f"-> Domain:   [{a_limit}, {b_limit}]")
    print(f"-> Nodes (n): {n_nodes}\n")

    print("Status: Warming up JAX JIT compiler...")
    try:
        _ = jit_audit(n_nodes).block_until_ready()
        print("Status: Compilation complete.\n")
    except Exception as e:
        print(f"⚠️ Compilation Error: {e}\n")
        return

    print("Status: Evaluating absolute precision limits...")
    try:
        quassian_ans = float(jit_audit(n_nodes))
        abs_error = abs(quassian_ans - true_integral)
        
        print(f"\n-> Verified Master Baseline : {true_integral:.15f}")
        print(f"-> Quassian Engine Output   : {quassian_ans:.15f}")
        print(f"-> Absolute Precision Error : {abs_error:.2e}")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return
    print("=" * 60)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    test_quassian_radar_chirp()
