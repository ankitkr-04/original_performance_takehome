"""
Mux vs Load tradeoff analysis - decide when to use arithmetic selection vs memory gather.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from problem import VLEN, HASH_STAGES


def analyze_mux_vs_load(
    tree_height: int = 10,
    batch_size: int = 256,
):
    """
    Calculate the cost of muxing (arithmetic node selection) vs loading (scatter-gather).
    
    This helps decide: for each tree level, should we preload nodes and use
    arithmetic selection, or just do scatter-gather?
    
    Mux cost depends on number of nodes to select from:
    - 2 nodes: 1 multiply_add (bit0 * diff + base)
    - 4 nodes: 2-level mux, ~4 valu ops
    - 8 nodes: 3-level mux, ~7 valu ops
    - N nodes: log2(N) levels, ~(3*log2(N) - 2) valu ops
    
    Load cost (scatter-gather):
    - VLEN loads per vector
    - 2 loads per cycle
    - Cost = VLEN / 2 = 4 cycles per vector
    - For group of 6: 24 cycles for loads, but can overlap with valu
    """
    n_vectors = batch_size // VLEN
    groups = (n_vectors + 5) // 6
    
    results = []
    
    for level in range(tree_height + 1):
        # Number of nodes at this level
        n_nodes = 2 ** level
        
        # Mux cost calculation
        if n_nodes == 1:
            mux_valu_ops = 0  # Just broadcast, no selection
            mux_description = "Broadcast only"
        elif n_nodes == 2:
            mux_valu_ops = 2  # bit = idx & 1; result = bit * diff + base
            mux_description = "1-level mux"
        elif n_nodes <= 4:
            mux_valu_ops = 8  # 2-level mux: compute bit0, bit1, two pair selects, final select
            mux_description = "2-level mux"
        elif n_nodes <= 8:
            mux_valu_ops = 14  # 3-level mux
            mux_description = "3-level mux"
        else:
            # General case: each level needs ~3 ops (compute bit, select pair, accumulate)
            levels = math.ceil(math.log2(n_nodes))
            mux_valu_ops = 3 * levels + 2  # Plus setup
            mux_description = f"{levels}-level mux"
        
        # Mux cycles (at 6 valu ops per cycle, processing 6 vectors)
        # For 6 vectors: mux_valu_ops per vector, all in parallel
        mux_cycles_per_group = math.ceil(mux_valu_ops / 6) * 6 // 6  # 1 cycle if <= 6 ops
        mux_cycles_per_group = max(1, mux_valu_ops // 6 + (1 if mux_valu_ops % 6 else 0))
        mux_total_cycles = mux_cycles_per_group * groups
        
        # Load cost (scatter-gather)
        # Each vector needs VLEN loads, 2 per cycle
        load_cycles_per_vector = VLEN // 2  # = 4
        load_cycles_per_group = load_cycles_per_vector * 6  # = 24 for 6 vectors
        load_total_cycles = load_cycles_per_group * groups
        
        # But loads can overlap with hash valu work!
        # Hash per group: ~15 valu ops
        # So effective load cost when overlapped: max(24, 15) = 24 cycles
        # But hash has to wait for load to complete, so true cost is ~24 + tail
        
        # Scratch cost for muxing
        mux_scratch = n_nodes * VLEN  # Need to store all node values
        
        # Winner
        if mux_total_cycles < load_total_cycles:
            winner = "MUX"
            savings = load_total_cycles - mux_total_cycles
        else:
            winner = "LOAD"
            savings = mux_total_cycles - load_total_cycles
        
        results.append({
            "level": level,
            "n_nodes": n_nodes,
            "mux_valu_ops": mux_valu_ops,
            "mux_cycles": mux_total_cycles,
            "mux_description": mux_description,
            "load_cycles": load_total_cycles,
            "mux_scratch": mux_scratch,
            "winner": winner,
            "savings": savings,
        })
    
    # Determine optimal strategy
    mux_levels = []
    load_levels = []
    total_scratch_for_mux = 0
    
    for r in results:
        if r["winner"] == "MUX" and total_scratch_for_mux + r["mux_scratch"] < 1000:  # Leave room
            mux_levels.append(r["level"])
            total_scratch_for_mux += r["mux_scratch"]
        else:
            load_levels.append(r["level"])
    
    return {
        "analysis": results,
        "recommended_mux_levels": mux_levels,
        "recommended_load_levels": load_levels,
        "mux_scratch_needed": total_scratch_for_mux,
        "summary": f"MUX levels 0-{max(mux_levels) if mux_levels else 0}, LOAD levels {min(load_levels) if load_levels else tree_height}+",
    }


def estimate_cycle_savings(
    current_cycles: int = 2512,
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Estimate how many cycles could be saved with optimal mux/load strategy.
    """
    analysis = analyze_mux_vs_load(tree_height, batch_size)
    
    # Current approach: mux levels 0,1,2, load levels 3+
    # Rounds 0-2 use levels 0,1,2 (indices 0, 1-2, 3-6)
    # Rounds 3+ use gather
    
    current_mux_levels = [0, 1, 2]
    current_load_rounds = max(0, rounds - 3)
    
    # Calculate current costs
    current_mux_cycles = sum(r["mux_cycles"] for r in analysis["analysis"] if r["level"] in current_mux_levels)
    current_load_cycles = current_load_rounds * analysis["analysis"][3]["load_cycles"]  # Approximate
    
    # Optimal would mux more levels (up to scratch limit)
    optimal_mux_levels = analysis["recommended_mux_levels"]
    optimal_load_start = max(optimal_mux_levels) + 1 if optimal_mux_levels else 0
    
    # But we can't mux indefinitely - rounds determine what we access
    # Round R accesses tree level R (roughly - it's more complex with index updates)
    
    potential_savings = {
        "current_mux_levels": current_mux_levels,
        "optimal_mux_levels": optimal_mux_levels,
        "current_approach": "MUX rounds 0-2, GATHER rounds 3+",
        "recommendation": f"MUX rounds 0-{len(optimal_mux_levels)-1}, GATHER rounds {len(optimal_mux_levels)}+",
        "estimated_savings": "Limited - main bottleneck is gather/hash interleaving, not mux strategy",
    }
    
    return potential_savings


if __name__ == "__main__":
    print("=== Mux vs Load Analysis ===")
    result = analyze_mux_vs_load()
    
    print("\nLevel | Nodes | Mux Ops | Mux Cycles | Load Cycles | Winner | Savings | Scratch")
    print("-" * 85)
    for r in result["analysis"][:8]:  # First 8 levels
        print(f"  {r['level']:3d} | {r['n_nodes']:5d} | {r['mux_valu_ops']:7d} | {r['mux_cycles']:10d} | {r['load_cycles']:11d} | {r['winner']:6s} | {r['savings']:7d} | {r['mux_scratch']:7d}")
    
    print(f"\nRecommended MUX levels: {result['recommended_mux_levels']}")
    print(f"Recommended LOAD levels: {result['recommended_load_levels']}")
    print(f"Total scratch for MUX: {result['mux_scratch_needed']}")
    
    print("\n=== Cycle Savings Estimate ===")
    savings = estimate_cycle_savings()
    for k, v in savings.items():
        print(f"  {k}: {v}")
