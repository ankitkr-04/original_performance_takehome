"""
Optimal Strategy Calculator - determines the best mux/gather strategy.

KEY INSIGHT: With tree_height=10 and rounds=16, traversal WRAPS at round 10.
This means rounds 10-15 access the same levels as rounds 0-5.

By muxing levels 0-5 (63 nodes), we only need gather for rounds 6-9!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import VLEN, SLOT_LIMITS, SCRATCH_SIZE


def calculate_optimal_strategy(
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Calculate the optimal mux strategy considering tree wrapping.
    
    Returns detailed breakdown of:
    - Which rounds access which tree levels
    - Optimal mux depth
    - Minimum achievable cycles
    """
    n_vectors = batch_size // VLEN
    # CORRECT: Tree has 2^(height+1) - 1 nodes, not 2^height - 1
    # Height=10 means 2047 nodes (levels 0-10, i.e. 11 levels total)
    tree_size = 2 ** (tree_height + 1) - 1
    
    # Determine which tree level each round accesses
    round_to_level = []
    idx = 0
    for r in range(rounds):
        # Calculate level from index
        level = 0
        temp = idx
        while temp > 0:
            temp = (temp - 1) // 2
            level += 1
        round_to_level.append(level)
        
        # Move to child (assuming even path for analysis)
        idx = 2 * idx + 1
        if idx >= tree_size:
            idx = 0  # Wrap to root
    
    # Find optimal mux depth
    # Muxing level L means we can skip gather for all rounds that access level <= L
    results = []
    
    for mux_depth in range(tree_height):
        # Nodes to preload for mux
        mux_nodes = sum(2**l for l in range(mux_depth + 1))
        mux_scratch = mux_nodes * VLEN
        
        # Count rounds that need gather (access level > mux_depth)
        gather_rounds = sum(1 for level in round_to_level if level > mux_depth)
        mux_rounds = rounds - gather_rounds
        
        # Calculate cycles
        init_cycles = 100 + mux_nodes  # Init + loading tree nodes
        
        # Mux round: ~60-80 cycles (valu only)
        mux_cycles_per = 60
        total_mux_cycles = mux_rounds * mux_cycles_per
        
        # Gather round: 128 cycles per round (memory-bound)
        gather_cycles_per = n_vectors * VLEN // SLOT_LIMITS['load']
        total_gather_cycles = gather_rounds * gather_cycles_per
        
        # Store
        store_cycles = n_vectors // SLOT_LIMITS['store']
        
        total_cycles = init_cycles + total_mux_cycles + total_gather_cycles + store_cycles
        
        scratch_feasible = mux_scratch < SCRATCH_SIZE * 0.8
        
        results.append({
            "mux_depth": mux_depth,
            "mux_levels": f"0-{mux_depth}",
            "mux_nodes": mux_nodes,
            "mux_scratch": mux_scratch,
            "scratch_feasible": scratch_feasible,
            "mux_rounds": mux_rounds,
            "gather_rounds": gather_rounds,
            "which_gather_rounds": [r for r, level in enumerate(round_to_level) if level > mux_depth],
            "init_cycles": init_cycles,
            "mux_cycles": total_mux_cycles,
            "gather_cycles": total_gather_cycles,
            "store_cycles": store_cycles,
            "total_cycles": total_cycles,
        })
    
    # Find optimal
    feasible = [r for r in results if r["scratch_feasible"]]
    optimal = min(feasible, key=lambda x: x["total_cycles"]) if feasible else results[0]
    
    return {
        "round_to_level": round_to_level,
        "tree_wraps_at_round": round_to_level.index(0, 1) if 0 in round_to_level[1:] else None,
        "all_strategies": results,
        "optimal_strategy": optimal,
        "current_strategy": {
            "description": "perf_takehome uses mux_depth=2 (levels 0-2, 7 nodes)",
            "gather_rounds": 13,
            "estimated_cycles": 2000,
        },
        "improvement_potential": {
            "current_to_optimal_savings": 2000 - optimal["total_cycles"],
            "key_insight": f"Mux levels 0-{optimal['mux_depth']} to reduce gather rounds from 13 to {optimal['gather_rounds']}",
        }
    }


def print_strategy_table():
    """Print a nice table of all strategies."""
    result = calculate_optimal_strategy()
    
    print("=== Round-to-Level Mapping ===")
    print(f"Rounds: {list(range(16))}")
    print(f"Levels: {result['round_to_level']}")
    print(f"Tree wraps at round: {result['tree_wraps_at_round']}")
    print()
    
    print("=== Strategy Comparison ===")
    print(f"{'Mux Depth':<10} {'Mux Nodes':<10} {'Scratch':<10} {'Mux Rnds':<10} {'Gather Rnds':<12} {'Cycles':<10} {'Feasible':<10}")
    print("-" * 82)
    
    for s in result["all_strategies"]:
        feasible = "YES" if s["scratch_feasible"] else "NO"
        optimal_marker = " <<< OPTIMAL" if s == result["optimal_strategy"] else ""
        print(f"{s['mux_depth']:<10} {s['mux_nodes']:<10} {s['mux_scratch']:<10} {s['mux_rounds']:<10} {s['gather_rounds']:<12} {s['total_cycles']:<10} {feasible}{optimal_marker}")
    
    print()
    print("=== Optimal Strategy Details ===")
    opt = result["optimal_strategy"]
    print(f"Mux levels: {opt['mux_levels']} ({opt['mux_nodes']} nodes)")
    print(f"Scratch usage: {opt['mux_scratch']} / {SCRATCH_SIZE} ({opt['mux_scratch']/SCRATCH_SIZE:.1%})")
    print(f"Gather rounds: {opt['which_gather_rounds']}")
    print(f"Estimated cycles: {opt['total_cycles']}")
    print()
    print(f"Key insight: {result['improvement_potential']['key_insight']}")


if __name__ == "__main__":
    print_strategy_table()
