#!/usr/bin/env python3
"""
Mux vs Load Tradeoff MCP - Calculates when to mux vs when to gather
===================================================================
Helps decide which tree levels should use arithmetic muxing vs memory loads.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS, VLEN, SCRATCH_SIZE


@dataclass
class LevelStrategy:
    """Strategy for a single tree level"""
    level: int
    nodes_at_level: int
    
    # Mux costs
    mux_valu_ops: int       # Number of VALU operations for mux
    mux_cycles: int         # Cycles to mux (ops / 6)
    mux_scratch_words: int  # Scratch needed for node values
    
    # Load costs  
    load_cycles: int        # Cycles to load (8 lanes / 2 per cycle = 4)
    addr_calc_cycles: int   # Cycles to compute addresses
    total_load_cycles: int  # load + addr calc
    
    # Recommendation
    recommended: str        # 'mux' or 'load'
    speedup: float          # How much faster is the recommended approach
    notes: str


@dataclass
class MuxLoadTradeoff:
    """Complete mux vs load tradeoff analysis"""
    levels: List[LevelStrategy]
    
    # Summary
    recommended_mux_depth: int
    recommended_gather_start: int
    total_mux_cycles: int
    total_gather_cycles: int
    
    # Constraints
    scratch_limit_depth: int  # Max depth before scratch overflow
    
    # Break-even analysis
    breakeven_level: int
    
    recommendations: List[str]


def calculate_mux_cost(level: int, num_parallel: int = 6) -> Tuple[int, int, int]:
    """
    Calculate cost to mux 1-of-N nodes at given level.
    
    Returns: (valu_ops, cycles, scratch_words_needed)
    
    Mux tree for N nodes:
    - 1 node (level 0): 0 ops, just broadcast
    - 2 nodes (level 1): 2 ops (compute bit, multiply_add)
    - 4 nodes (level 2): 8 ops (2-bit selection tree)
    - 8 nodes (level 3): ~20 ops (3-bit selection)
    - General: ~6*level ops for 2^level nodes
    """
    nodes = 2 ** level
    
    if level == 0:
        # Just broadcast - 1 vbroadcast per group
        return 1, 1, VLEN
    elif level == 1:
        # bit = idx & 1, result = bit * diff + base
        # 2 ops per element, 6 elements -> 2 instructions
        return 2 * num_parallel, 2, 3 * VLEN  # 2 nodes + diff
    elif level == 2:
        # Need 2-bit mux: ~8-10 ops per element
        # idx-3, bit0, bit1, pair0, pair1, diff, result
        return 8 * num_parallel, 8, 6 * VLEN  # 4 nodes + 2 diffs
    else:
        # General case: log2(N) levels of mux
        # Each level needs ~3 ops (bit extract, multiply_add for 2 pairs, combine)
        # Total: ~3 * level * num_parallel ops
        ops = 3 * level * num_parallel
        cycles = math.ceil(ops / 6)
        # Need all nodes + diffs at each level
        scratch = nodes * VLEN + (nodes - 1) * VLEN  # nodes + intermediate diffs
        return ops, cycles, scratch


def calculate_load_cost(num_parallel: int = 6) -> Tuple[int, int, int]:
    """
    Calculate cost to gather node values via memory loads.
    
    Returns: (load_ops, cycles, addr_calc_cycles)
    
    For each vector:
    - 8 scalar loads (gather), 2 per cycle = 4 cycles per vector
    - Address calculation: 2 VALU ops (broadcast + add)
    
    For num_parallel vectors in a group:
    - 8 * num_parallel loads = 4 * num_parallel load cycles
    - But we can overlap with hash from previous group!
    """
    loads_per_vector = VLEN
    loads_per_group = loads_per_vector * num_parallel
    load_cycles = math.ceil(loads_per_group / 2)  # 2 loads per cycle
    
    addr_calc_cycles = 2  # vbroadcast + add, done in parallel for group
    
    return loads_per_group, load_cycles, addr_calc_cycles


def analyze_mux_load_tradeoff(
    tree_height: int = 10,
    batch_size: int = 256,
    num_parallel: int = 6,
    hash_cycles_per_group: int = 18  # Typical hash computation cycles
) -> MuxLoadTradeoff:
    """
    Analyze mux vs load tradeoff for each tree level.
    
    Key insight: When gather is pipelined with hash from previous group,
    the effective cost is max(gather_cycles, hash_cycles) not the sum.
    """
    
    n_vectors = batch_size // VLEN
    n_groups = math.ceil(n_vectors / num_parallel)
    
    levels = []
    total_scratch_for_mux = 0
    
    _, load_cycles, addr_calc_cycles = calculate_load_cost(num_parallel)
    
    # With pipelining, effective gather time is max(load_time, hash_time)
    effective_load_cycles = max(load_cycles + addr_calc_cycles, hash_cycles_per_group)
    
    scratch_limit_depth = 0
    breakeven_level = -1
    
    for level in range(min(tree_height, 10)):  # Analyze up to level 10
        nodes = 2 ** level
        
        mux_ops, mux_cycles_per_group, mux_scratch = calculate_mux_cost(level, num_parallel)
        
        # Total mux cycles for all groups (no pipelining benefit for mux-only rounds)
        total_mux = mux_cycles_per_group * n_groups + hash_cycles_per_group * n_groups
        
        # Total load cycles (with pipelining)
        # First group: addr_calc + load + hash (no overlap)
        # Subsequent groups: max(load, hash) due to overlap
        first_group = addr_calc_cycles + load_cycles + hash_cycles_per_group
        other_groups = effective_load_cycles * (n_groups - 1) if n_groups > 1 else 0
        total_load = first_group + other_groups
        
        # Check scratch constraint
        total_scratch_for_mux += mux_scratch
        if total_scratch_for_mux < SCRATCH_SIZE * 0.7:  # Leave 30% headroom
            scratch_limit_depth = level
        
        # Determine recommendation
        if mux_cycles_per_group < load_cycles:
            recommended = 'mux'
            speedup = load_cycles / mux_cycles_per_group if mux_cycles_per_group > 0 else float('inf')
        else:
            recommended = 'load'
            speedup = mux_cycles_per_group / load_cycles if load_cycles > 0 else 1.0
            if breakeven_level < 0:
                breakeven_level = level
        
        notes = ""
        if level <= 2:
            notes = "Easy to mux with arithmetic selection"
        elif level <= 4:
            notes = "Muxing still feasible but complex"
        else:
            notes = "Gathering more efficient at this depth"
        
        levels.append(LevelStrategy(
            level=level,
            nodes_at_level=nodes,
            mux_valu_ops=mux_ops,
            mux_cycles=mux_cycles_per_group,
            mux_scratch_words=mux_scratch,
            load_cycles=load_cycles,
            addr_calc_cycles=addr_calc_cycles,
            total_load_cycles=load_cycles + addr_calc_cycles,
            recommended=recommended,
            speedup=speedup,
            notes=notes
        ))
    
    # Find recommended depths
    recommended_mux_depth = 0
    for lvl in levels:
        if lvl.recommended == 'mux' and lvl.level <= scratch_limit_depth:
            recommended_mux_depth = lvl.level
    
    recommended_gather_start = recommended_mux_depth + 1
    
    # Calculate total cycles for each strategy
    total_mux_cycles = sum(lvl.mux_cycles * n_groups for lvl in levels[:recommended_mux_depth+1])
    total_gather_cycles = effective_load_cycles * n_groups * (tree_height - recommended_gather_start)
    
    recommendations = [
        f"Mux rounds 0-{recommended_mux_depth} (levels with ≤{2**recommended_mux_depth} nodes)",
        f"Gather from round {recommended_gather_start} onwards",
        f"Break-even point: Level {breakeven_level} ({2**breakeven_level} nodes)",
        f"Scratch constraint allows preloading up to level {scratch_limit_depth}",
    ]
    
    if recommended_mux_depth < 2:
        recommendations.append(
            "⚠ Consider preloading more levels - muxing is very cheap for small node counts"
        )
    
    return MuxLoadTradeoff(
        levels=levels,
        recommended_mux_depth=recommended_mux_depth,
        recommended_gather_start=recommended_gather_start,
        total_mux_cycles=total_mux_cycles,
        total_gather_cycles=total_gather_cycles,
        scratch_limit_depth=scratch_limit_depth,
        breakeven_level=breakeven_level,
        recommendations=recommendations
    )


def format_mux_load_tradeoff(analysis: MuxLoadTradeoff) -> str:
    """Format tradeoff analysis as readable report."""
    lines = [
        "=" * 70,
        "MUX vs LOAD TRADEOFF ANALYSIS",
        "=" * 70,
        "",
        "┌" + "─" * 68 + "┐",
        "│  Level │  Nodes │ Mux Cyc │ Load Cyc │ Recommend │   Speedup │",
        "├" + "─" * 68 + "┤",
    ]
    
    for lvl in analysis.levels:
        marker = "✓" if lvl.recommended == 'mux' else " "
        lines.append(
            f"│ {marker} {lvl.level:4d} │ {lvl.nodes_at_level:6d} │ {lvl.mux_cycles:7d} │ {lvl.total_load_cycles:8d} │ {lvl.recommended:9s} │ {lvl.speedup:8.2f}x │"
        )
    
    lines.extend([
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│                         SUMMARY                                   │",
        "├" + "─" * 68 + "┤",
        f"│ Recommended Mux Depth:      Level 0-{analysis.recommended_mux_depth}                            │",
        f"│ Start Gathering at:         Round {analysis.recommended_gather_start}                              │",
        f"│ Break-even Level:           {analysis.breakeven_level}                                    │",
        f"│ Scratch Limit:              Level {analysis.scratch_limit_depth}                              │",
        "└" + "─" * 68 + "┘",
        ""
    ])
    
    lines.append("RECOMMENDATIONS:")
    for i, rec in enumerate(analysis.recommendations, 1):
        lines.append(f"  {i}. {rec}")
    
    lines.extend([
        "",
        "COST COMPARISON (per group of 6 vectors):",
        f"  • 1-of-2 mux (level 1):  ~2 cycles",
        f"  • 1-of-4 mux (level 2):  ~8 cycles", 
        f"  • Gather (any level):    ~26 cycles (but can pipeline)",
        f"  • Hash computation:      ~18 cycles (overlaps with gather)",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    analysis = analyze_mux_load_tradeoff(
        tree_height=10,
        batch_size=256,
        num_parallel=6
    )
    print(format_mux_load_tradeoff(analysis))
