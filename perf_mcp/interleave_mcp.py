"""
Interleave analysis - understand how well loads and valus overlap.
Shows the actual instruction pattern during gather phases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, Input, build_mem_image, VLEN


def analyze_interleave(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    show_instructions: int = 50,
    start_from: int = 0,
):
    """
    Analyze how well load and valu operations are interleaved.
    
    Shows actual instruction patterns to understand where interleaving breaks down.
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
    
    # Analyze patterns
    patterns = {
        "load_only": [],
        "valu_only": [],
        "combined": [],
        "store_only": [],
        "other": [],
    }
    
    # Find runs of same pattern type
    runs = []
    current_run = None
    current_start = 0
    
    for i, instr in enumerate(instrs):
        has_load = 'load' in instr
        has_valu = 'valu' in instr
        has_store = 'store' in instr
        
        if has_load and has_valu:
            pattern = "combined"
        elif has_load and not has_valu:
            pattern = "load_only"
        elif has_valu and not has_load:
            pattern = "valu_only"
        elif has_store:
            pattern = "store"
        else:
            pattern = "other"
        
        if pattern != current_run:
            if current_run is not None:
                runs.append({
                    "pattern": current_run,
                    "start": current_start,
                    "end": i - 1,
                    "length": i - current_start,
                })
            current_run = pattern
            current_start = i
    
    # Final run
    if current_run is not None:
        runs.append({
            "pattern": current_run,
            "start": current_start,
            "end": len(instrs) - 1,
            "length": len(instrs) - current_start,
        })
    
    # Find longest runs by type
    longest_runs = {}
    for run in runs:
        p = run["pattern"]
        if p not in longest_runs or run["length"] > longest_runs[p]["length"]:
            longest_runs[p] = run
    
    # Count total by pattern
    pattern_counts = {"combined": 0, "load_only": 0, "valu_only": 0, "store": 0, "other": 0}
    for run in runs:
        pattern_counts[run["pattern"]] += run["length"]
    
    # Show actual instructions in a range
    instruction_sample = []
    end_idx = min(start_from + show_instructions, len(instrs))
    for i in range(start_from, end_idx):
        instr = instrs[i]
        summary = []
        if 'load' in instr:
            loads = instr['load']
            load_types = [op[0] for op in loads]
            summary.append(f"L:{len(loads)}({','.join(set(load_types))})")
        if 'valu' in instr:
            valus = instr['valu']
            valu_ops = [op[0] for op in valus]
            summary.append(f"V:{len(valus)}({','.join(set(valu_ops))})")
        if 'alu' in instr:
            alus = instr['alu']
            summary.append(f"A:{len(alus)}")
        if 'store' in instr:
            stores = instr['store']
            summary.append(f"S:{len(stores)}")
        if 'flow' in instr:
            summary.append("FLOW")
        
        instruction_sample.append(f"{i:4d}: {' | '.join(summary)}")
    
    # Find where interleaving opportunities are missed
    missed_opportunities = []
    for i in range(len(instrs) - 1):
        curr = instrs[i]
        next_i = instrs[i + 1]
        
        # Load-only followed by valu-only (or vice versa) = missed opportunity
        curr_load_only = 'load' in curr and 'valu' not in curr
        curr_valu_only = 'valu' in curr and 'load' not in curr
        next_load_only = 'load' in next_i and 'valu' not in next_i
        next_valu_only = 'valu' in next_i and 'load' not in next_i
        
        if (curr_load_only and next_valu_only) or (curr_valu_only and next_load_only):
            missed_opportunities.append(i)
    
    return {
        "total_instructions": len(instrs),
        "pattern_counts": pattern_counts,
        "interleave_ratio": round(pattern_counts["combined"] / len(instrs) * 100, 1),
        "longest_runs": longest_runs,
        "num_runs": len(runs),
        "missed_opportunities": len(missed_opportunities),
        "first_10_missed": missed_opportunities[:10],
        "instruction_sample": instruction_sample,
        "sample_range": f"{start_from}-{end_idx-1}",
    }


def find_gather_phases(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Identify where gather phases occur and analyze their efficiency.
    
    A gather phase is a sequence with scatter-gather loads (not const/vload).
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
    
    gather_phases = []
    current_phase = None
    
    for i, instr in enumerate(instrs):
        if 'load' not in instr:
            if current_phase is not None:
                current_phase["end"] = i - 1
                current_phase["length"] = current_phase["end"] - current_phase["start"] + 1
                gather_phases.append(current_phase)
                current_phase = None
            continue
        
        loads = instr['load']
        # Check if any load is a scatter-gather (not const, vload, or vbroadcast)
        is_gather = any(op[0] == 'load' for op in loads)
        
        if is_gather:
            if current_phase is None:
                current_phase = {
                    "start": i,
                    "combined_count": 0,
                    "load_only_count": 0,
                    "total_loads": 0,
                    "total_valus": 0,
                }
            
            has_valu = 'valu' in instr
            if has_valu:
                current_phase["combined_count"] += 1
                current_phase["total_valus"] += len(instr['valu'])
            else:
                current_phase["load_only_count"] += 1
            current_phase["total_loads"] += len(loads)
        else:
            if current_phase is not None:
                current_phase["end"] = i - 1
                current_phase["length"] = current_phase["end"] - current_phase["start"] + 1
                gather_phases.append(current_phase)
                current_phase = None
    
    # Final phase
    if current_phase is not None:
        current_phase["end"] = len(instrs) - 1
        current_phase["length"] = current_phase["end"] - current_phase["start"] + 1
        gather_phases.append(current_phase)
    
    # Calculate efficiency for each phase
    for phase in gather_phases:
        total = phase["combined_count"] + phase["load_only_count"]
        phase["interleave_pct"] = round(phase["combined_count"] / total * 100, 1) if total > 0 else 0
    
    # Summary
    total_gather_instrs = sum(p["length"] for p in gather_phases)
    total_combined = sum(p["combined_count"] for p in gather_phases)
    
    return {
        "num_gather_phases": len(gather_phases),
        "total_gather_instructions": total_gather_instrs,
        "total_combined_in_gather": total_combined,
        "gather_interleave_ratio": round(total_combined / total_gather_instrs * 100, 1) if total_gather_instrs > 0 else 0,
        "phases": gather_phases[:10],  # First 10 phases
        "phase_lengths": [p["length"] for p in gather_phases],
    }


if __name__ == "__main__":
    print("=== Interleave Analysis: perf_takehome ===")
    result = analyze_interleave("perf_takehome", show_instructions=30, start_from=480)
    print(f"Pattern counts: {result['pattern_counts']}")
    print(f"Interleave ratio: {result['interleave_ratio']}%")
    print(f"Longest runs: {result['longest_runs']}")
    print(f"Missed opportunities: {result['missed_opportunities']}")
    print(f"\nSample instructions ({result['sample_range']}):")
    for line in result['instruction_sample'][:20]:
        print(f"  {line}")
    
    print("\n=== Gather Phase Analysis: perf_takehome ===")
    result = find_gather_phases("perf_takehome")
    print(f"Num gather phases: {result['num_gather_phases']}")
    print(f"Gather interleave ratio: {result['gather_interleave_ratio']}%")
    print(f"Phase lengths: {result['phase_lengths'][:10]}")
    
    print("\n=== Interleave Analysis: takehome_diff ===")
    result = analyze_interleave("takehome_diff", show_instructions=30, start_from=480)
    print(f"Pattern counts: {result['pattern_counts']}")
    print(f"Interleave ratio: {result['interleave_ratio']}%")
    print(f"Missed opportunities: {result['missed_opportunities']}")
    
    print("\n=== Gather Phase Analysis: takehome_diff ===")
    result = find_gather_phases("takehome_diff")
    print(f"Num gather phases: {result['num_gather_phases']}")
    print(f"Gather interleave ratio: {result['gather_interleave_ratio']}%")
    print(f"Phase lengths: {result['phase_lengths'][:10]}")
