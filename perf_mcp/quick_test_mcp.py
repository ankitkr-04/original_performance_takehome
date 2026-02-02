"""
Quick kernel test tool - run a kernel and get pass/fail + metrics in one call.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Machine, Tree, Input, build_mem_image, reference_kernel2, VLEN


def quick_test(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    seed: int = 123,
):
    """
    Run a kernel and return pass/fail + key metrics.
    
    Args:
        kernel_module: "perf_takehome" or "takehome_diff"
        tree_height: Height of the tree
        rounds: Number of rounds
        batch_size: Batch size
        seed: Random seed
    
    Returns:
        Dict with test results and metrics
    """
    # Import the requested kernel module
    if kernel_module == "perf_takehome":
        from perf_takehome import KernelBuilder, do_kernel_test
    elif kernel_module == "takehome_diff":
        from takehome_diff import KernelBuilder, do_kernel_test
    else:
        return {"error": f"Unknown kernel module: {kernel_module}"}
    
    random.seed(seed)
    forest = Tree.generate(tree_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    n_vectors = batch_size // VLEN
    
    # Count instruction types
    total_instrs = len(kb.instrs)
    load_only = sum(1 for i in kb.instrs if 'load' in i and 'valu' not in i)
    valu_only = sum(1 for i in kb.instrs if 'valu' in i and 'load' not in i)
    combined = sum(1 for i in kb.instrs if 'load' in i and 'valu' in i)
    store_instrs = sum(1 for i in kb.instrs if 'store' in i)
    
    # Run the kernel
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=1, trace=False)
    machine.prints = False
    
    passed = True
    error_round = None
    error_msg = None
    
    try:
        for i, ref_mem in enumerate(reference_kernel2(mem, {})):
            machine.run()
            inp_values_p = ref_mem[6]
            actual = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            expected = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            if actual != expected:
                passed = False
                error_round = i
                # Find first mismatch
                for j, (a, e) in enumerate(zip(actual, expected)):
                    if a != e:
                        error_msg = f"Round {i}: mismatch at index {j}, got {a}, expected {e}"
                        break
                break
    except Exception as e:
        passed = False
        error_msg = str(e)
    
    cycles = machine.cycle
    
    # Calculate efficiency metrics
    interleave_ratio = combined / total_instrs * 100 if total_instrs > 0 else 0
    
    # Theoretical minimum calculation (with optimal mux strategy)
    # KEY INSIGHT: With tree_height=10 and rounds=16, tree wraps at round 10
    # Rounds 10-15 access same levels as rounds 0-5
    # Optimal: mux levels 0-6 (127 nodes), only gather for rounds 7-9
    n_vectors = batch_size // 8  # VLEN = 8
    gather_cycles_per_round = n_vectors * 8 // 2  # 128 cycles
    
    # Current kernel (mux 0-2): 13 gather rounds
    current_strategy_min = 130 + 3 * 100 + 13 * gather_cycles_per_round + 16
    
    # Optimal strategy (mux 0-6): only 3 gather rounds
    optimal_strategy_min = 130 + 13 * 60 + 3 * gather_cycles_per_round + 16
    
    # Use current strategy as baseline (what kernel is likely doing)
    theoretical_min_cycles = current_strategy_min
    
    return {
        "passed": passed,
        "cycles": cycles,
        "error_round": error_round,
        "error_msg": error_msg,
        "total_instructions": total_instrs,
        "load_only": load_only,
        "valu_only": valu_only,
        "combined_load_valu": combined,
        "store_instructions": store_instrs,
        "interleave_ratio": round(interleave_ratio, 1),
        "theoretical_min_current": theoretical_min_cycles,
        "theoretical_min_optimal": optimal_strategy_min,
        "efficiency_vs_current": round(theoretical_min_cycles / cycles * 100, 1) if cycles > 0 else 0,
        "efficiency_vs_optimal": round(optimal_strategy_min / cycles * 100, 1) if cycles > 0 else 0,
        "scratch_used": kb.scratch_ptr,
        "n_vectors": n_vectors,
        "optimization_hint": "Use mux levels 0-6 to reduce gather rounds from 13 to 3" if cycles > 1500 else "Good progress!",
    }


def compare_kernels(
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    seed: int = 123,
):
    """
    Compare perf_takehome vs takehome_diff side by side.
    """
    result_a = quick_test("perf_takehome", tree_height, rounds, batch_size, seed)
    result_b = quick_test("takehome_diff", tree_height, rounds, batch_size, seed)
    
    return {
        "perf_takehome": result_a,
        "takehome_diff": result_b,
        "cycle_diff": result_b["cycles"] - result_a["cycles"] if result_b.get("cycles") and result_a.get("cycles") else None,
        "interleave_diff": round(result_b.get("interleave_ratio", 0) - result_a.get("interleave_ratio", 0), 1),
        "winner": "takehome_diff" if result_b.get("cycles", float('inf')) < result_a.get("cycles", float('inf')) else "perf_takehome",
    }


if __name__ == "__main__":
    print("=== Quick Test: perf_takehome ===")
    result = quick_test("perf_takehome")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    print("\n=== Quick Test: takehome_diff ===")
    result = quick_test("takehome_diff")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    print("\n=== Comparison ===")
    comp = compare_kernels()
    print(f"  Cycle diff: {comp['cycle_diff']}")
    print(f"  Interleave diff: {comp['interleave_diff']}%")
    print(f"  Winner: {comp['winner']}")
