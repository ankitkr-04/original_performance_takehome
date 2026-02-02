"""
CrossRoundPipelineAnalyzer - Measure round-transition bubbles.

The key insight: To hit sub-1400 cycles, rounds must BLEED into each other.
Most kernels have "Drain" phase (finishing hash) and "Init" phase (calculating addresses).
This tool finds where the Load engine is idle waiting for VALU to finish.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, Machine, VLEN, SLOT_LIMITS


def analyze_round_transitions(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Analyze the pipeline efficiency at round transitions.
    
    Returns:
    - Round boundaries in the instruction stream
    - "Bubble" cycles where one engine is idle at transitions
    - Overlap opportunities between rounds
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
    
    # Find round boundaries by looking for patterns
    # Typically: pause instructions, or sudden changes in register usage
    
    # Analyze each instruction for engine activity
    engine_activity = []
    for idx, instr in enumerate(instrs):
        has_load = 'load' in instr and any(op[0] in ['load', 'load_offset'] for op in instr.get('load', []))
        has_valu = 'valu' in instr and len(instr.get('valu', [])) > 0
        has_alu = 'alu' in instr and len(instr.get('alu', [])) > 0
        has_store = 'store' in instr and len(instr.get('store', [])) > 0
        
        engine_activity.append({
            'idx': idx,
            'has_load': has_load,
            'has_valu': has_valu,
            'has_alu': has_alu,
            'has_store': has_store,
            'load_count': sum(1 for op in instr.get('load', []) if op[0] in ['load', 'load_offset']),
            'valu_count': len(instr.get('valu', [])),
        })
    
    # Find "bubbles" - runs of instructions where Load is idle but VALU is busy
    # These are transition inefficiencies
    bubbles = []
    current_bubble = None
    
    for i, act in enumerate(engine_activity):
        # Load idle, VALU busy = potential bubble
        is_valu_only = act['has_valu'] and not act['has_load']
        # VALU idle, Load busy = also potential bubble  
        is_load_only = act['has_load'] and not act['has_valu']
        
        if is_valu_only:
            if current_bubble is None or current_bubble['type'] != 'valu_only':
                if current_bubble:
                    bubbles.append(current_bubble)
                current_bubble = {'type': 'valu_only', 'start': i, 'length': 1}
            else:
                current_bubble['length'] += 1
        elif is_load_only:
            if current_bubble is None or current_bubble['type'] != 'load_only':
                if current_bubble:
                    bubbles.append(current_bubble)
                current_bubble = {'type': 'load_only', 'start': i, 'length': 1}
            else:
                current_bubble['length'] += 1
        else:
            if current_bubble:
                bubbles.append(current_bubble)
                current_bubble = None
    
    if current_bubble:
        bubbles.append(current_bubble)
    
    # Filter to significant bubbles (> 5 cycles)
    significant_bubbles = [b for b in bubbles if b['length'] > 5]
    
    # Calculate total bubble cycles
    total_valu_only = sum(b['length'] for b in bubbles if b['type'] == 'valu_only')
    total_load_only = sum(b['length'] for b in bubbles if b['type'] == 'load_only')
    total_interleaved = len(instrs) - total_valu_only - total_load_only
    
    # Find round transition points (where we see gather pattern start)
    # Gather pattern: burst of load-only or load+valu with scatter-gather loads
    transitions = []
    gather_starts = []
    
    # Look for sequences of 8+ consecutive loads (scatter-gather pattern)
    load_run = 0
    for i, act in enumerate(engine_activity):
        if act['load_count'] > 0:
            if load_run == 0:
                gather_start_candidate = i
            load_run += act['load_count']
        else:
            if load_run >= VLEN * n_vectors // 4:  # Significant gather
                gather_starts.append({
                    'start': gather_start_candidate,
                    'total_loads': load_run,
                    'estimated_round': len(gather_starts) + 3  # Rounds 3+ have gathers
                })
            load_run = 0
    
    # Analyze gaps between gathers (these are the round transitions)
    for i in range(len(gather_starts) - 1):
        end_of_current = gather_starts[i]['start'] + gather_starts[i]['total_loads'] // 2
        start_of_next = gather_starts[i + 1]['start']
        gap = start_of_next - end_of_current
        
        # What's in the gap?
        gap_valu_only = sum(1 for j in range(end_of_current, start_of_next) 
                           if j < len(engine_activity) and 
                           engine_activity[j]['has_valu'] and not engine_activity[j]['has_load'])
        
        transitions.append({
            'from_round': gather_starts[i]['estimated_round'],
            'to_round': gather_starts[i + 1]['estimated_round'],
            'gap_cycles': gap,
            'valu_only_in_gap': gap_valu_only,
            'potential_overlap': min(gap_valu_only, 20),  # Could start next round's addr calc
        })
    
    # Calculate overlap ratio
    overlap_ratio = total_interleaved / len(instrs) * 100 if instrs else 0
    
    # Potential savings
    potential_savings = sum(t['potential_overlap'] for t in transitions)
    
    # SUCCESS PARADOX: When fully optimized, gather phases merge into one continuous stream
    # This is IDEAL behavior, not a problem!
    continuous_pipeline = len(gather_starts) <= 3  # Ideal: mux levels 0-6, only gather 7,8,9
    
    return {
        "total_instructions": len(instrs),
        "interleaved_cycles": total_interleaved,
        "valu_only_cycles": total_valu_only,
        "load_only_cycles": total_load_only,
        "overlap_ratio": round(overlap_ratio, 1),
        "significant_bubbles": significant_bubbles[:10],  # Top 10
        "round_transitions": transitions,
        "gather_phases_detected": len(gather_starts),
        "gather_starts": gather_starts,
        "potential_savings_from_overlap": potential_savings,
        # SUCCESS PARADOX DETECTION
        "continuous_pipeline_achieved": continuous_pipeline,
        "pipeline_quality_note": (
            "SUCCESS PARADOX: Only 1-3 gather phases detected. This is IDEAL! "
            "The kernel has achieved a continuous global pipeline where loads/hash/stores "
            "flow without round boundaries. No further round-overlap optimization needed."
        ) if continuous_pipeline else (
            f"Found {len(gather_starts)} distinct gather phases. "
            "Consider merging rounds by pre-computing addresses or muxing tree levels."
        ),
        "recommendations": [
            f"Total VALU-only bubbles: {total_valu_only} cycles - these could overlap with next round's address calc",
            f"Total Load-only bubbles: {total_load_only} cycles - fill with hash computation",
            f"Potential savings from round overlap: ~{potential_savings} cycles",
        ] if potential_savings > 50 and not continuous_pipeline else (
            ["✓ Continuous pipeline achieved - round transitions are seamless"] if continuous_pipeline
            else ["Round transitions are well-optimized"]
        ),
    }


def calculate_round_overlap(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Measure how many instructions contain operations from two different "phases".
    
    Perfect pipelining = high overlap count.
    Sequential rounds = low overlap count.
    """
    result = analyze_round_transitions(kernel_module, tree_height, rounds, batch_size)
    
    # Overlap score: interleaved / total
    overlap_score = result['interleaved_cycles'] / result['total_instructions'] * 100
    
    # Target: >60% for good pipelining
    quality = "EXCELLENT" if overlap_score > 60 else "GOOD" if overlap_score > 45 else "POOR"
    
    return {
        "overlap_score": round(overlap_score, 1),
        "quality": quality,
        "interleaved_instructions": result['interleaved_cycles'],
        "total_instructions": result['total_instructions'],
        "bubble_summary": {
            "valu_only": result['valu_only_cycles'],
            "load_only": result['load_only_cycles'],
        },
        "improvement_hint": "Start next round's address calculation while finishing current round's hash" 
                          if quality == "POOR" else "Good overlap, focus on other optimizations",
    }


if __name__ == "__main__":
    print("=== Cross-Round Pipeline Analysis: perf_takehome ===")
    result = analyze_round_transitions("perf_takehome")
    print(f"Total instructions: {result['total_instructions']}")
    print(f"Interleaved: {result['interleaved_cycles']} ({result['overlap_ratio']}%)")
    print(f"VALU-only bubbles: {result['valu_only_cycles']}")
    print(f"Load-only bubbles: {result['load_only_cycles']}")
    print(f"\nSignificant bubbles (>5 cycles):")
    for b in result['significant_bubbles'][:5]:
        print(f"  {b['type']} at cycle {b['start']}: {b['length']} cycles")
    print(f"\nRound transitions:")
    for t in result['round_transitions'][:5]:
        print(f"  Round {t['from_round']}→{t['to_round']}: gap={t['gap_cycles']}, overlap potential={t['potential_overlap']}")
    print(f"\nRecommendations:")
    for r in result['recommendations']:
        print(f"  - {r}")
    
    print("\n=== Round Overlap Score ===")
    overlap = calculate_round_overlap("perf_takehome")
    print(f"Overlap score: {overlap['overlap_score']}% ({overlap['quality']})")
