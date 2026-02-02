"""
What-If Calculator - estimate impact of hypothetical changes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import VLEN, SLOT_LIMITS


def what_if(
    current_cycles: int = 2512,
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    # Hypothetical changes
    loads_per_cycle: int = None,  # Default is 2
    valus_per_cycle: int = None,  # Default is 6
    mux_rounds: int = None,       # Default is 3 (rounds 0-2)
    perfect_interleave: bool = False,
):
    """
    Calculate estimated cycles under hypothetical changes.
    
    Use this to understand:
    - "What if I could do 4 loads per cycle?"
    - "What if I muxed 5 rounds instead of 3?"
    - "What if I had perfect interleaving?"
    """
    n_vectors = batch_size // VLEN
    groups = (n_vectors + 5) // 6
    
    actual_loads = loads_per_cycle or SLOT_LIMITS['load']
    actual_valus = valus_per_cycle or SLOT_LIMITS['valu']
    actual_mux_rounds = mux_rounds if mux_rounds is not None else 3
    
    # Calculate per-phase costs
    
    # Init: ~100-130 cycles (doesn't change much)
    init_cycles = 127
    
    # Mux rounds (no gather needed)
    # ~20 valu ops per group, 6 ops per cycle
    valu_per_mux_round = groups * 20 / actual_valus * 6  # Approx
    mux_round_cycles = actual_mux_rounds * valu_per_mux_round
    
    # Gather rounds
    gather_rounds = rounds - actual_mux_rounds
    
    # Per gather round:
    # - Address calc: 2 instructions per group
    # - Gather: n_vectors * VLEN loads / loads_per_cycle
    # - Hash: ~18 valu instructions per group
    
    gather_loads = n_vectors * VLEN
    gather_cycles_per_round = gather_loads / actual_loads
    hash_cycles_per_round = groups * 18 / actual_valus * 6
    
    if perfect_interleave:
        # Perfect overlap: max(gather, hash)
        per_gather_round = max(gather_cycles_per_round, hash_cycles_per_round)
    else:
        # Current: ~70% interleave
        overlap = 0.7
        per_gather_round = gather_cycles_per_round + hash_cycles_per_round * (1 - overlap)
    
    gather_total = gather_rounds * per_gather_round
    
    # Store: ~32 cycles
    store_cycles = n_vectors * 2 / SLOT_LIMITS['store']
    
    estimated_total = int(init_cycles + mux_round_cycles + gather_total + store_cycles)
    
    # Calculate savings
    savings = current_cycles - estimated_total
    speedup = current_cycles / estimated_total if estimated_total > 0 else 0
    
    return {
        "estimated_cycles": estimated_total,
        "savings_vs_current": savings,
        "speedup": round(speedup, 2),
        "breakdown": {
            "init": int(init_cycles),
            "mux_rounds": int(mux_round_cycles),
            "gather_rounds": int(gather_total),
            "store": int(store_cycles),
        },
        "parameters": {
            "loads_per_cycle": actual_loads,
            "valus_per_cycle": actual_valus,
            "mux_rounds": actual_mux_rounds,
            "gather_rounds": gather_rounds,
            "perfect_interleave": perfect_interleave,
        },
        "feasibility": "POSSIBLE" if estimated_total > 1000 else "VERY_AGGRESSIVE",
    }


def explore_optimizations(current_cycles: int = 2512):
    """
    Explore various optimization scenarios.
    """
    scenarios = []
    
    # Baseline
    base = what_if(current_cycles)
    scenarios.append(("Baseline (current constraints)", base))
    
    # More mux rounds
    for mux in [4, 5, 6]:
        result = what_if(current_cycles, mux_rounds=mux)
        scenarios.append((f"Mux {mux} rounds (instead of 3)", result))
    
    # Perfect interleaving
    result = what_if(current_cycles, perfect_interleave=True)
    scenarios.append(("Perfect load/valu interleaving", result))
    
    # More loads per cycle (hypothetical HW change)
    for loads in [3, 4]:
        result = what_if(current_cycles, loads_per_cycle=loads)
        scenarios.append((f"{loads} loads/cycle (HW change)", result))
    
    # Combined: more mux + perfect interleave
    result = what_if(current_cycles, mux_rounds=5, perfect_interleave=True)
    scenarios.append(("Mux 5 rounds + perfect interleave", result))
    
    return scenarios


def format_what_if_report(scenarios):
    """Format exploration results."""
    lines = [
        "=" * 70,
        "WHAT-IF ANALYSIS",
        "=" * 70,
        "",
        f"{'Scenario':<45} {'Cycles':>8} {'Savings':>8} {'Speedup':>8}",
        "-" * 70,
    ]
    
    for name, result in scenarios:
        lines.append(
            f"{name:<45} {result['estimated_cycles']:>8} "
            f"{result['savings_vs_current']:>8} {result['speedup']:>7}x"
        )
    
    lines.extend([
        "-" * 70,
        "",
        "KEY INSIGHTS:",
    ])
    
    # Find best achievable
    best = min(scenarios, key=lambda s: s[1]['estimated_cycles'])
    lines.append(f"  Best scenario: {best[0]} -> {best[1]['estimated_cycles']} cycles")
    
    # Check if 1300 is achievable
    if best[1]['estimated_cycles'] > 1300:
        lines.append(f"  Target 1300: NOT achievable with these optimizations")
        lines.append(f"  Minimum possible: ~{best[1]['estimated_cycles']} cycles")
        lines.append(f"  To reach 1300: Need algorithmic breakthrough (different approach)")
    else:
        lines.append(f"  Target 1300: ACHIEVABLE with {best[0]}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    scenarios = explore_optimizations(2512)
    print(format_what_if_report(scenarios))
