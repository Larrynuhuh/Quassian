import time
import numpy as np
from gauss_legendre import integrate
import jax
import jax.numpy as jnp
def test_quassian():
    print("=" * 60)
    print("          QUASSIAN QUADRATURE PERFORMANCE TEST          ")
    print("=" * 60)

    # 1. Define a complex, high-frequency oscillatory test function
    # Integrating cos(50 * x) over [0, 1]
    # The exact mathematical analytical integral is sin(50)/50
    test_freq = 50.0
    def func(x):
        return jnp.cos(test_freq * x)
    
    a_limit = 0.0
    b_limit = 1.0
    true_integral = float((np.sin(test_freq) - np.sin(0.0)) / test_freq)

    # We select a high n = 500 to evaluate the engine where GW begins to lag
    n_nodes = 20000

    print(f"Test Setup:")
    print(f"-> Function: f(x) = cos({test_freq} * x)")
    print(f"-> Domain:   [{a_limit}, {b_limit}]")
    print(f"-> Nodes (n): {n_nodes}\n")

    # 2. Warm up the JAX JIT compiler for the integration routine
    print("Status: Warming up JAX JIT compiler for Quassian...")
    try:
        # We run it once to compile the execution graph
        _ = integrate(func, a_limit, b_limit, n_nodes)
        print("Status: Compilation complete.\n")
    except Exception as e:
        print(f"⚠️ Compilation Error encountered: {e}")
        print("This is likely where the 'failure' shows its teeth. Let's see the numbers anyway.\n")

    # 3. Accuracy Evaluation
    try:
        quassian_ans = float(integrate(func, a_limit, b_limit, n_nodes))
        abs_error = abs(quassian_ans - true_integral)
        
        print(f"-> True Analytical Answer: {true_integral:.15f}")
        print(f"-> Quassian Engine Output:  {quassian_ans:.15f}")
        print(f"-> Absolute Precision Error: {abs_error:.2e}")
    except Exception as e:
        print(f"❌ Execution failed during evaluation: {e}")
        return

    print("-" * 60)

    # 4. Speed Benchmarking (Running multiple loops)
    iterations = 10
    print(f"Status: Clocking Quassian over {iterations} loops...")
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        res = integrate(func, a_limit, b_limit, n_nodes)
    
    # Block async dispatch to catch true hardware time
    res.block_until_ready()
    t1 = time.perf_counter()

    total_time_ms = (t1 - t0) * 1000
    avg_time_ms = total_time_ms / iterations

    print(f"-> Total time for {iterations} loops: {total_time_ms:.2f} ms")
    print(f"-> Average Execution Speed:       {avg_time_ms:.4f} ms per integration 🚀")
    print("=" * 60)

if __name__ == "__main__":
 # Calls your Biprox if both are in file, or modify to test_quassian()
    test_quassian()
