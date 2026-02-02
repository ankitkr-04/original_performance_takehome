"""
Bottleneck identifier - analyze a kernel and identify THE primary bottleneck.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, VLEN, SLOT_LIMITS


def identify_bottleneck(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Analyze a kernel and identify the primary bottleneck.
    
    Returns a diagnosis with:
    - The bottleneck type (load bandwidth, valu throughput, interleaving, etc.)
    - Evidence for the diagnosis
    - Specific recommendations
    """
    if kernel_module == "perf_takehome":
        from perf_takehome import KernelBuilder
    elif kernel_module == "takehome_diff":
        from takehome_diff import KernelBuilder
    else:
        return {"error": f"Unknown kernel: {kernel_module}"}
    
    random.seed(123)
    forest = Tree.generate(tree_height)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), batch_size, rounds)
    
    instrs = kb.instrs
    n_vectors = batch_size // VLEN
    
    # Analyze instruction patterns
    total = len(instrs)
    load_only = sum(1 for i in instrs if 'load' in i and 'valu' not in i)
    valu_only = sum(1 for i in instrs if 'valu' in i and 'load' not in i)
    combined = sum(1 for i in instrs if 'load' in i and 'valu' in i)
    store_only = sum(1 for i in instrs if 'store' in i)
    
    # Count actual operations
    total_loads = sum(len(i.get('load', [])) for i in instrs)
    total_valus = sum(len(i.get('valu', [])) for i in instrs)
    total_stores = sum(len(i.get('store', [])) for i in instrs)
    
    # Calculate utilization
    load_util = total_loads / (total * SLOT_LIMITS['load']) * 100
    valu_util = total_valus / (total * SLOT_LIMITS['valu']) * 100
    
    # Interleave ratio
    interleave_ratio = combined / total * 100 if total > 0 else 0
    
    # Find longest single-engine runs
    longest_load_run = 0
    longest_valu_run = 0
    current_load_run = 0
    current_valu_run = 0
    
    for instr in instrs:
        is_load_only = 'load' in instr and 'valu' not in instr
        is_valu_only = 'valu' in instr and 'load' not in instr
        
        if is_load_only:
            current_load_run += 1
            longest_load_run = max(longest_load_run, current_load_run)
        else:
            current_load_run = 0
        
        if is_valu_only:
            current_valu_run += 1
            longest_valu_run = max(longest_valu_run, current_valu_run)
        else:
            current_valu_run = 0
    
    # Diagnose the bottleneck
    bottlenecks = []
    
    # Check interleaving
    if interleave_ratio < 40:
        bottlenecks.append({
            "type": "POOR_INTERLEAVING",
            "severity": "HIGH",
            "evidence": f"Only {interleave_ratio:.1f}% of instructions combine load+valu",
            "impact": f"{load_only + valu_only} wasted instruction slots",
            "fix": "Restructure emit_gather_with_hash to better interleave operations",
        })
    
    # Check for long single-engine runs
    if longest_valu_run > 100:
        bottlenecks.append({
            "type": "VALU_ONLY_RUN",
            "severity": "MEDIUM",
            "evidence": f"Longest valu-only run: {longest_valu_run} instructions",
            "impact": f"Load engine idle during these cycles",
            "fix": "These are likely rounds 0-2 (no gather). Consider preloading more tree levels.",
        })
    
    if longest_load_run > 20:
        bottlenecks.append({
            "type": "LOAD_ONLY_RUN",
            "severity": "MEDIUM",
            "evidence": f"Longest load-only run: {longest_load_run} instructions",
            "impact": f"VALU engine idle during these cycles",
            "fix": "Interleave hash computation with gather operations",
        })
    
    # Check load bandwidth
    gather_rounds = max(0, rounds - 3)
    expected_gather_loads = gather_rounds * n_vectors * VLEN
    actual_gather_cycles = expected_gather_loads / 2  # 2 loads per cycle
    
    if load_util < 50:
        bottlenecks.append({
            "type": "LOW_LOAD_UTILIZATION",
            "severity": "LOW",
            "evidence": f"Load utilization only {load_util:.1f}%",
            "impact": "Memory bandwidth underutilized",
            "fix": "Not a problem if valu-bound",
        })
    
    # Determine primary bottleneck
    if bottlenecks:
        primary = max(bottlenecks, key=lambda b: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[b["severity"]])
    else:
        primary = {
            "type": "WELL_OPTIMIZED",
            "severity": "LOW",
            "evidence": f"Good interleaving ({interleave_ratio:.1f}%), balanced utilization",
            "impact": "Near theoretical minimum",
            "fix": "Consider algorithmic changes for further improvement",
        }
    
    # Calculate potential savings
    if interleave_ratio < 48:  # Compare to perf_takehome's 48.6%
        potential_savings = int((48 - interleave_ratio) / 100 * total)
    else:
        potential_savings = 0
    
    return {
        "primary_bottleneck": primary,
        "all_bottlenecks": bottlenecks,
        "metrics": {
            "total_instructions": total,
            "load_only": load_only,
            "valu_only": valu_only,
            "combined": combined,
            "interleave_ratio": round(interleave_ratio, 1),
            "load_utilization": round(load_util, 1),
            "valu_utilization": round(valu_util, 1),
            "longest_valu_run": longest_valu_run,
            "longest_load_run": longest_load_run,
        },
        "potential_cycle_savings": potential_savings,
        "recommendation": primary["fix"],
    }


def format_bottleneck_report(analysis):
    """Format bottleneck analysis as readable text."""
    lines = [
        "=" * 60,
        "BOTTLENECK ANALYSIS REPORT",
        "=" * 60,
        "",
        "PRIMARY BOTTLENECK:",
        f"  Type: {analysis['primary_bottleneck']['type']}",
        f"  Severity: {analysis['primary_bottleneck']['severity']}",
        f"  Evidence: {analysis['primary_bottleneck']['evidence']}",
        f"  Impact: {analysis['primary_bottleneck']['impact']}",
        f"  Fix: {analysis['primary_bottleneck']['fix']}",
        "",
        "METRICS:",
    ]
    
    for k, v in analysis['metrics'].items():
        lines.append(f"  {k}: {v}")
    
    lines.extend([
        "",
        f"POTENTIAL SAVINGS: ~{analysis['potential_cycle_savings']} cycles",
        "",
        "ALL BOTTLENECKS:",
    ])
    
    for b in analysis['all_bottlenecks']:
        lines.append(f"  [{b['severity']}] {b['type']}: {b['evidence']}")
    
    lines.extend([
        "",
        "=" * 60,
        f"RECOMMENDATION: {analysis['recommendation']}",
        "=" * 60,
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Bottleneck Analysis: perf_takehome ===")
    analysis = identify_bottleneck("perf_takehome")
    print(format_bottleneck_report(analysis))
    
    print("\n" + "=" * 60 + "\n")
    
    print("=== Bottleneck Analysis: takehome_diff ===")
    analysis = identify_bottleneck("takehome_diff")
    print(format_bottleneck_report(analysis))
