"""
Per-round instruction breakdown - see exactly what each round generates.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, VLEN, HASH_STAGES


def analyze_round_instructions(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Break down instruction count by round.
    
    Estimates round boundaries based on instruction patterns.
    """
    if kernel_module == "perf_takehome":
        from perf_takehome import KernelBuilder
    elif kernel_module == "takehome_diff":
        from takehome_diff import KernelBuilder
    else:
        return {"error": f"Unknown kernel module: {kernel_module}"}
    
    random.seed(123)
    forest = Tree.generate(tree_height)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), batch_size, rounds)
    
    instrs = kb.instrs
    n_vectors = batch_size // VLEN
    
    # Find the "pause" instructions - they mark boundaries
    pause_indices = []
    for i, instr in enumerate(instrs):
        if 'flow' in instr:
            for op in instr['flow']:
                if op[0] == 'pause':
                    pause_indices.append(i)
    
    # Estimate round boundaries by looking for patterns
    # Each round processes all vectors, so we look for repeating patterns
    
    # Count loads and valus per instruction
    load_counts = []
    valu_counts = []
    for instr in instrs:
        load_counts.append(len(instr.get('load', [])))
        valu_counts.append(len(instr.get('valu', [])))
    
    # Identify init phase (before first pause or before heavy valu work)
    init_end = pause_indices[0] if pause_indices else 100
    
    # Group instructions by examining valu density
    # Rounds 0-2 are valu-heavy with no scatter-gather loads
    # Rounds 3+ have scatter-gather loads interleaved with valu
    
    phases = []
    
    # Init phase
    phases.append({
        "name": "init",
        "start": 0,
        "end": init_end,
        "instructions": init_end + 1,
        "has_gather": False,
    })
    
    # Main computation - estimate based on expected pattern
    # Each round for n_vectors=32, group_size=6 has ~6 groups
    # Round 0-2: ~20 valu instrs per group = ~120 per round
    # Round 3+: ~24 gather instrs + ~18 hash instrs (overlapped) = ~24-30 per group
    
    main_start = init_end + 1
    
    # Find store phase (at the end)
    store_start = None
    for i in range(len(instrs) - 1, -1, -1):
        if 'store' in instrs[i]:
            if store_start is None:
                store_start = i
        elif store_start is not None:
            store_start = i + 1
            break
    
    if store_start is None:
        store_start = len(instrs)
    
    main_end = store_start - 1
    main_instructions = main_end - main_start + 1
    
    # Estimate instructions per round
    # Rounds 0-2: valu only, ~120 instrs each
    # Rounds 3+: gather + hash, varies
    
    valu_only_rounds = min(3, rounds)
    gather_rounds = max(0, rounds - 3)
    
    estimated_per_valu_round = 120
    estimated_per_gather_round = (main_instructions - valu_only_rounds * estimated_per_valu_round) // gather_rounds if gather_rounds > 0 else 0
    
    phases.append({
        "name": "main_computation",
        "start": main_start,
        "end": main_end,
        "instructions": main_instructions,
        "valu_only_rounds": valu_only_rounds,
        "gather_rounds": gather_rounds,
        "est_per_valu_round": estimated_per_valu_round,
        "est_per_gather_round": estimated_per_gather_round,
    })
    
    # Store phase
    phases.append({
        "name": "store",
        "start": store_start,
        "end": len(instrs) - 1,
        "instructions": len(instrs) - store_start,
    })
    
    # Count scatter-gather loads in main phase
    scatter_gather_loads = 0
    for i in range(main_start, main_end + 1):
        instr = instrs[i]
        if 'load' in instr:
            for op in instr['load']:
                if op[0] == 'load':  # scatter-gather load
                    scatter_gather_loads += 1
    
    # Expected scatter-gather loads: gather_rounds * n_vectors * VLEN / 2 (2 per cycle)
    expected_sg_loads = gather_rounds * n_vectors * VLEN
    
    return {
        "total_instructions": len(instrs),
        "phases": phases,
        "pause_indices": pause_indices,
        "scatter_gather_loads": scatter_gather_loads,
        "expected_sg_loads": expected_sg_loads,
        "sg_load_efficiency": round(expected_sg_loads / max(1, scatter_gather_loads) * 100, 1),
        "n_vectors": n_vectors,
        "valu_only_rounds": valu_only_rounds,
        "gather_rounds": gather_rounds,
    }


def calculate_theoretical_minimum(
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Calculate the theoretical minimum cycles for the kernel.
    
    Based on:
    - Init: Loading constants, broadcasting
    - Rounds 0-2: Valu-only (6 slots), no memory bottleneck
    - Rounds 3+: Memory-bound (2 loads/cycle) or valu-bound (6 slots/cycle)
    - Store: 2 stores/cycle
    """
    n_vectors = batch_size // VLEN
    groups = (n_vectors + 5) // 6  # Ceiling division
    
    # Init phase estimate
    # - Load ~20 constants
    # - Broadcast ~15 vectors
    # - Compute base addresses for n_vectors
    init_cycles = 20 + 15 + n_vectors * 2
    
    # Rounds 0-2: Valu only
    # Per round: Each vector needs hash computation but no gather
    # ~15 valu ops per vector with multiply_add optimization
    # At 6 valu slots per cycle: n_vectors * 15 / 6 = ~80 cycles per round
    valu_only_rounds = min(3, rounds)
    valu_only_cycles = valu_only_rounds * (n_vectors * 15 // 6)
    
    # Rounds 3+: Gather + Hash overlapped
    # CORRECTED: Each gather round needs n_vectors * VLEN = 256 loads
    # At 2 loads/cycle = 128 cycles per gather round
    # Hash can overlap, so gather is the bottleneck
    gather_rounds = max(0, rounds - 3)
    gather_cycles_per_round = n_vectors * VLEN // 2  # 128 cycles
    
    # With perfect interleaving, hash overlaps completely
    # Without perfect interleaving, add ~20% overhead
    gather_cycles = gather_rounds * gather_cycles_per_round
    
    # Store phase
    # 2 stores per cycle, n_vectors stores
    store_cycles = n_vectors // 2
    
    # Total theoretical minimum
    theoretical_min = init_cycles + valu_only_cycles + gather_cycles + store_cycles
    
    # More aggressive estimate (perfect scheduling)
    # Init can be reduced with clever constant reuse
    # Gather is the hard floor: 13 rounds * 128 cycles = 1664 cycles
    perfect_min = (
        100 +  # Init (aggressive)
        valu_only_rounds * 80 +  # Valu rounds (aggressive)
        gather_rounds * gather_cycles_per_round +  # Gather rounds (memory-bound)
        16  # Store
    )
    
    return {
        "init_cycles": init_cycles,
        "valu_only_cycles": valu_only_cycles,
        "gather_cycles": gather_cycles,
        "store_cycles": store_cycles,
        "theoretical_minimum": theoretical_min,
        "perfect_minimum": perfect_min,
        "breakdown": {
            "n_vectors": n_vectors,
            "groups": groups,
            "gather_cycles_per_round": gather_cycles_per_round,
            "valu_only_rounds": valu_only_rounds,
            "gather_rounds": gather_rounds,
        },
        "target_1300_feasibility": "POSSIBLE" if perfect_min < 1300 else f"IMPOSSIBLE_WITHOUT_ALGO_CHANGE (min ~{perfect_min})",
        "note": "1363 cycles achieved by Claude suggests algorithmic breakthrough (fewer gather rounds?)"
    }


if __name__ == "__main__":
    print("=== Per-Round Analysis: perf_takehome ===")
    result = analyze_round_instructions("perf_takehome")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    print("\n=== Per-Round Analysis: takehome_diff ===")
    result = analyze_round_instructions("takehome_diff")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    print("\n=== Theoretical Minimum ===")
    result = calculate_theoretical_minimum()
    for k, v in result.items():
        print(f"  {k}: {v}")
