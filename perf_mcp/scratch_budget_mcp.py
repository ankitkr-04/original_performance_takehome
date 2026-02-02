#!/usr/bin/env python3
"""
Scratch Budget MCP - Analyzes scratch memory usage and capacity
===============================================================
Helps understand how much scratch space is available for preloading tree nodes,
muxing constants, and temporary buffers.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SCRATCH_SIZE, VLEN, SLOT_LIMITS


@dataclass
class ScratchAllocation:
    """Info about a scratch memory allocation"""
    name: str
    start_addr: int
    length: int
    category: str  # 'data', 'constant', 'buffer', 'temp'


@dataclass 
class ScratchBudget:
    """Complete scratch budget analysis"""
    total_size: int
    peak_usage: int
    remaining: int
    utilization_pct: float
    
    # Breakdown by category
    data_bytes: int        # indices, values arrays
    constant_bytes: int    # hash constants, preloaded nodes
    buffer_bytes: int      # node buffers, address buffers
    temp_bytes: int        # temporary computation space
    
    # Capacity analysis
    max_preload_depth: int      # How many tree levels can be preloaded
    max_mux_nodes: int          # How many nodes fit in scratch for muxing
    vectors_per_group: int      # Current vectors processed per group
    
    # Recommendations
    warning: str
    suggestions: List[str]
    
    allocations: List[ScratchAllocation]


def analyze_scratch_budget(
    batch_size: int = 256,
    tree_height: int = 10,
    rounds: int = 16,
    current_preload_depth: int = 2,  # rounds 0,1,2 use preloaded
    num_parallel: int = 6
) -> ScratchBudget:
    """
    Analyze scratch memory budget and capacity.
    
    Args:
        batch_size: Number of items in batch (256)
        tree_height: Height of tree (10)
        rounds: Number of rounds (16)
        current_preload_depth: Current tree depth being preloaded
        num_parallel: Vectors processed in parallel
        
    Returns:
        ScratchBudget with complete analysis
    """
    n_vectors = batch_size // VLEN  # 32 vectors
    
    allocations = []
    addr = 0
    
    def alloc(name: str, length: int, category: str) -> int:
        nonlocal addr
        allocations.append(ScratchAllocation(name, addr, length, category))
        start = addr
        addr += length
        return start
    
    # === Core data structures ===
    # Scalar temps (8 words)
    alloc("tmp_scalars", 8, "temp")
    
    # Vector indices and values (batch_size * 2)
    for vi in range(n_vectors):
        alloc(f"indices_{vi}", VLEN, "data")
    for vi in range(n_vectors):
        alloc(f"values_{vi}", VLEN, "data")
    
    # Base addresses for indices/values
    for vi in range(n_vectors):
        alloc(f"v_idx_base_{vi}", 1, "data")
        alloc(f"v_val_base_{vi}", 1, "data")
    
    # === Constants ===
    # Hash constants (6 stages * 2 values each)
    alloc("hash_constants", 6 * 2 * VLEN, "constant")
    
    # Common constants (v_one, v_two, v_three, v_n_nodes)
    alloc("v_one", VLEN, "constant")
    alloc("v_two", VLEN, "constant")
    alloc("v_three", VLEN, "constant")
    alloc("v_n_nodes", VLEN, "constant")
    
    # === Preloaded tree nodes ===
    # Round 0: node 0 (1 node)
    # Round 1: nodes 1,2 (2 nodes) + diff
    # Round 2: nodes 3,4,5,6 (4 nodes) + 2 diffs
    preload_nodes = 0
    if current_preload_depth >= 0:
        alloc("v_node_r0", VLEN, "constant")
        preload_nodes += 1
    if current_preload_depth >= 1:
        alloc("v_node_r1_1", VLEN, "constant")
        alloc("v_node_r1_2", VLEN, "constant")
        alloc("v_node_r1_diff", VLEN, "constant")
        preload_nodes += 3  # 2 nodes + 1 diff
    if current_preload_depth >= 2:
        for i in range(4):
            alloc(f"v_node_r2_{i}", VLEN, "constant")
        alloc("v_r2_diff01", VLEN, "constant")
        alloc("v_r2_diff23", VLEN, "constant")
        preload_nodes += 6  # 4 nodes + 2 diffs
    
    # === Working buffers ===
    # Node buffers A and B for double-buffering
    for i in range(num_parallel):
        alloc(f"v_node_A_{i}", VLEN, "buffer")
    for i in range(num_parallel):
        alloc(f"v_node_B_{i}", VLEN, "buffer")
    
    # Address buffers A and B
    for i in range(num_parallel):
        alloc(f"v_addr_A_{i}", VLEN, "buffer")
    for i in range(num_parallel):
        alloc(f"v_addr_B_{i}", VLEN, "buffer")
    
    # Temp vectors for computation
    for i in range(num_parallel):
        alloc(f"v_tmp1_{i}", VLEN, "temp")
    for i in range(num_parallel):
        alloc(f"v_tmp2_{i}", VLEN, "temp")
    
    # === Calculate totals ===
    peak_usage = addr
    remaining = SCRATCH_SIZE - peak_usage
    
    data_bytes = sum(a.length for a in allocations if a.category == 'data')
    constant_bytes = sum(a.length for a in allocations if a.category == 'constant')
    buffer_bytes = sum(a.length for a in allocations if a.category == 'buffer')
    temp_bytes = sum(a.length for a in allocations if a.category == 'temp')
    
    # === Capacity calculations ===
    # How many more tree levels could we preload?
    # Level N has 2^N nodes, each needs VLEN words
    available_for_preload = remaining
    max_additional_depth = 0
    cumulative_nodes = 0
    for level in range(current_preload_depth + 1, tree_height):
        nodes_at_level = 2 ** level
        space_needed = nodes_at_level * VLEN
        if cumulative_nodes * VLEN + space_needed <= available_for_preload:
            cumulative_nodes += nodes_at_level
            max_additional_depth = level - current_preload_depth
        else:
            break
    
    max_preload_depth = current_preload_depth + max_additional_depth
    max_mux_nodes = remaining // VLEN
    
    # === Warnings and suggestions ===
    utilization_pct = (peak_usage / SCRATCH_SIZE) * 100
    
    if utilization_pct > 95:
        warning = "CRITICAL: Over 95% scratch utilization - very tight!"
    elif utilization_pct > 90:
        warning = "WARNING: Over 90% scratch utilization - limited headroom"
    elif utilization_pct > 80:
        warning = "CAUTION: Over 80% scratch utilization"
    else:
        warning = "OK: Healthy scratch budget"
    
    suggestions = []
    
    if max_preload_depth > current_preload_depth:
        nodes_at_next = 2 ** (current_preload_depth + 1)
        suggestions.append(
            f"Can preload {max_additional_depth} more levels (up to level {max_preload_depth}). "
            f"Next level ({current_preload_depth + 1}) has {nodes_at_next} nodes = {nodes_at_next * VLEN} words."
        )
    
    if remaining >= VLEN * 4:
        suggestions.append(
            f"Room for {remaining // VLEN} more vector registers "
            f"({remaining} words). Consider precomputing more constants."
        )
    
    if num_parallel < 6 and remaining >= VLEN * 8:
        suggestions.append(
            "Could increase num_parallel to 6 for better VALU utilization."
        )
    
    return ScratchBudget(
        total_size=SCRATCH_SIZE,
        peak_usage=peak_usage,
        remaining=remaining,
        utilization_pct=utilization_pct,
        data_bytes=data_bytes,
        constant_bytes=constant_bytes,
        buffer_bytes=buffer_bytes,
        temp_bytes=temp_bytes,
        max_preload_depth=max_preload_depth,
        max_mux_nodes=max_mux_nodes,
        vectors_per_group=num_parallel,
        warning=warning,
        suggestions=suggestions,
        allocations=allocations
    )


def format_scratch_budget(budget: ScratchBudget) -> str:
    """Format scratch budget as readable report."""
    lines = [
        "=" * 70,
        "SCRATCH BUDGET ANALYSIS",
        "=" * 70,
        "",
        f"Total Scratch Size:    {budget.total_size:5d} words",
        f"Peak Usage:            {budget.peak_usage:5d} words ({budget.utilization_pct:.1f}%)",
        f"Remaining:             {budget.remaining:5d} words",
        "",
        "┌" + "─" * 68 + "┐",
        "│                      BREAKDOWN BY CATEGORY                        │",
        "├" + "─" * 68 + "┤",
        f"│ Data (indices/values):    {budget.data_bytes:5d} words  ({100*budget.data_bytes/budget.total_size:5.1f}%)            │",
        f"│ Constants (hash/nodes):   {budget.constant_bytes:5d} words  ({100*budget.constant_bytes/budget.total_size:5.1f}%)            │",
        f"│ Buffers (node/addr):      {budget.buffer_bytes:5d} words  ({100*budget.buffer_bytes/budget.total_size:5.1f}%)            │",
        f"│ Temps (computation):      {budget.temp_bytes:5d} words  ({100*budget.temp_bytes/budget.total_size:5.1f}%)            │",
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│                      CAPACITY ANALYSIS                            │",
        "├" + "─" * 68 + "┤",
        f"│ Max Preload Depth:        Level {budget.max_preload_depth}                               │",
        f"│ Max Mux Nodes:            {budget.max_mux_nodes:5d} vectors                           │",
        f"│ Vectors Per Group:        {budget.vectors_per_group:5d}                                  │",
        "└" + "─" * 68 + "┘",
        "",
        f"⚠ {budget.warning}",
        ""
    ]
    
    if budget.suggestions:
        lines.append("SUGGESTIONS:")
        for i, sug in enumerate(budget.suggestions, 1):
            lines.append(f"  {i}. {sug}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    budget = analyze_scratch_budget(
        batch_size=256,
        tree_height=10,
        rounds=16,
        current_preload_depth=2,
        num_parallel=6
    )
    print(format_scratch_budget(budget))
