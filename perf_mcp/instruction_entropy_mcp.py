#!/usr/bin/env python3
"""
Instruction Entropy MCP - Analyzes slot competition and engine bottlenecks
=========================================================================
Shows when engines are oversubscribed (wanted more slots than available)
and identifies the true bottleneck limiting throughput.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS, VLEN


@dataclass
class CycleBottleneck:
    """Bottleneck info for a single cycle"""
    cycle: int
    engine: str
    used: int
    limit: int
    wasted_potential: int  # How many more ops we could do if limit was higher


@dataclass
class EngineStats:
    """Statistics for one engine"""
    name: str
    max_slots: int
    
    total_ops: int
    avg_utilization: float
    peak_utilization: float
    
    saturated_cycles: int       # Cycles where all slots used
    underutilized_cycles: int   # Cycles where < 50% slots used
    idle_cycles: int            # Cycles with 0 usage
    
    # Histogram of usage
    usage_histogram: Dict[int, int]  # slots_used -> count


@dataclass 
class InstructionEntropy:
    """Complete instruction entropy analysis"""
    total_cycles: int
    
    # Per-engine stats
    engine_stats: Dict[str, EngineStats]
    
    # Bottleneck analysis
    primary_bottleneck: str
    secondary_bottleneck: str
    
    # Competition analysis
    load_valu_competition: int    # Cycles where both wanted more
    concurrent_saturation: int    # Cycles where multiple engines saturated
    
    # Packing efficiency
    avg_slots_per_cycle: float
    max_possible_slots: int
    packing_efficiency: float
    
    # Worst cycles
    worst_bottleneck_cycles: List[CycleBottleneck]
    
    # Phase analysis
    init_phase_cycles: int
    compute_phase_cycles: int
    store_phase_cycles: int
    
    recommendations: List[str]


def analyze_instruction_entropy(instrs: List[dict]) -> InstructionEntropy:
    """
    Analyze instruction stream for slot competition and bottlenecks.
    """
    
    engine_usage = defaultdict(lambda: defaultdict(int))  # engine -> cycle -> usage
    
    for cycle, instr in enumerate(instrs):
        for engine, slots in instr.items():
            if engine == 'debug':
                continue
            engine_usage[engine][cycle] = len(slots)
    
    # Calculate per-engine stats
    engine_stats = {}
    
    for engine, limit in SLOT_LIMITS.items():
        if engine == 'debug':
            continue
            
        usage = engine_usage.get(engine, {})
        
        total_ops = sum(usage.values())
        
        # Calculate utilizations
        usages = [usage.get(c, 0) for c in range(len(instrs))]
        avg_util = (sum(usages) / (len(instrs) * limit)) * 100 if instrs else 0
        peak_util = (max(usages) / limit) * 100 if usages else 0
        
        saturated = sum(1 for u in usages if u >= limit)
        underutilized = sum(1 for u in usages if 0 < u < limit * 0.5)
        idle = sum(1 for u in usages if u == 0)
        
        # Usage histogram
        histogram = defaultdict(int)
        for u in usages:
            histogram[u] += 1
        
        engine_stats[engine] = EngineStats(
            name=engine,
            max_slots=limit,
            total_ops=total_ops,
            avg_utilization=avg_util,
            peak_utilization=peak_util,
            saturated_cycles=saturated,
            underutilized_cycles=underutilized,
            idle_cycles=idle,
            usage_histogram=dict(histogram)
        )
    
    # Find primary bottleneck (most saturated engine)
    bottleneck_scores = {
        name: stats.saturated_cycles / len(instrs) if instrs else 0
        for name, stats in engine_stats.items()
    }
    sorted_bottlenecks = sorted(bottleneck_scores.items(), key=lambda x: -x[1])
    
    primary_bottleneck = sorted_bottlenecks[0][0] if sorted_bottlenecks else "none"
    secondary_bottleneck = sorted_bottlenecks[1][0] if len(sorted_bottlenecks) > 1 else "none"
    
    # Competition analysis
    load_valu_competition = 0
    concurrent_saturation = 0
    
    for cycle in range(len(instrs)):
        load_sat = engine_usage['load'].get(cycle, 0) >= SLOT_LIMITS['load']
        valu_sat = engine_usage['valu'].get(cycle, 0) >= SLOT_LIMITS['valu']
        
        if load_sat and valu_sat:
            concurrent_saturation += 1
        elif load_sat or valu_sat:
            # One is saturated, check if other wanted more
            if load_sat and engine_usage['valu'].get(cycle, 0) > 0:
                load_valu_competition += 1
            elif valu_sat and engine_usage['load'].get(cycle, 0) > 0:
                load_valu_competition += 1
    
    # Packing efficiency
    total_slots = sum(
        engine_usage[e].get(c, 0) 
        for c in range(len(instrs)) 
        for e in SLOT_LIMITS if e != 'debug'
    )
    max_possible = sum(SLOT_LIMITS[e] for e in SLOT_LIMITS if e != 'debug') * len(instrs)
    avg_slots = total_slots / len(instrs) if instrs else 0
    packing_eff = (total_slots / max_possible * 100) if max_possible else 0
    
    # Find worst bottleneck cycles
    worst_cycles = []
    for cycle in range(len(instrs)):
        for engine, limit in SLOT_LIMITS.items():
            if engine == 'debug':
                continue
            used = engine_usage[engine].get(cycle, 0)
            if used >= limit:
                worst_cycles.append(CycleBottleneck(
                    cycle=cycle,
                    engine=engine,
                    used=used,
                    limit=limit,
                    wasted_potential=0  # Hard to calculate without knowing what was queued
                ))
    
    # Sort by cycle and take first 20
    worst_cycles.sort(key=lambda x: x.cycle)
    worst_cycles = worst_cycles[:20]
    
    # Phase analysis (rough heuristic)
    init_phase = 0
    store_phase = 0
    
    for cycle in range(min(150, len(instrs))):
        if engine_usage['load'].get(cycle, 0) > 0 and engine_usage['valu'].get(cycle, 0) == 0:
            init_phase += 1
    
    for cycle in range(max(0, len(instrs) - 50), len(instrs)):
        if engine_usage['store'].get(cycle, 0) > 0:
            store_phase += 1
    
    compute_phase = len(instrs) - init_phase - store_phase
    
    # Generate recommendations
    recommendations = []
    
    if primary_bottleneck == 'load':
        recommendations.append(
            f"Load engine is primary bottleneck ({engine_stats['load'].saturated_cycles} saturated cycles). "
            "Consider: preloading more data, muxing instead of gathering, or spreading loads."
        )
    elif primary_bottleneck == 'valu':
        recommendations.append(
            f"VALU engine is primary bottleneck ({engine_stats['valu'].saturated_cycles} saturated cycles). "
            "This is often good! Consider if any ops can move to ALU."
        )
    
    if concurrent_saturation > len(instrs) * 0.3:
        recommendations.append(
            f"High concurrent saturation ({concurrent_saturation} cycles with both load and valu saturated). "
            "Good pipelining! Further optimization requires algorithmic changes."
        )
    
    if packing_eff < 30:
        recommendations.append(
            f"Low packing efficiency ({packing_eff:.1f}%). "
            "Many instruction slots are unused. Consider batching more operations per cycle."
        )
    
    valu_stats = engine_stats.get('valu', None)
    if valu_stats and valu_stats.idle_cycles > len(instrs) * 0.3:
        recommendations.append(
            f"VALU idle for {valu_stats.idle_cycles} cycles ({100*valu_stats.idle_cycles/len(instrs):.1f}%). "
            "Consider interleaving hash computation with loads."
        )
    
    return InstructionEntropy(
        total_cycles=len(instrs),
        engine_stats=engine_stats,
        primary_bottleneck=primary_bottleneck,
        secondary_bottleneck=secondary_bottleneck,
        load_valu_competition=load_valu_competition,
        concurrent_saturation=concurrent_saturation,
        avg_slots_per_cycle=avg_slots,
        max_possible_slots=sum(SLOT_LIMITS[e] for e in SLOT_LIMITS if e != 'debug'),
        packing_efficiency=packing_eff,
        worst_bottleneck_cycles=worst_cycles,
        init_phase_cycles=init_phase,
        compute_phase_cycles=compute_phase,
        store_phase_cycles=store_phase,
        recommendations=recommendations
    )


def format_instruction_entropy(analysis: InstructionEntropy) -> str:
    """Format entropy analysis as readable report."""
    lines = [
        "=" * 70,
        "INSTRUCTION ENTROPY & SLOT COMPETITION ANALYSIS",
        "=" * 70,
        "",
        f"Total Instructions: {analysis.total_cycles}",
        "",
        "┌" + "─" * 68 + "┐",
        "│                    ENGINE UTILIZATION                             │",
        "├" + "─" * 68 + "┤",
        "│ Engine │   Limit │  Avg Util │ Saturated │    Idle │ Total Ops │",
        "├" + "─" * 68 + "┤",
    ]
    
    for name in ['load', 'valu', 'alu', 'store']:
        if name in analysis.engine_stats:
            s = analysis.engine_stats[name]
            lines.append(
                f"│ {name:6s} │ {s.max_slots:7d} │ {s.avg_utilization:8.1f}% │ {s.saturated_cycles:9d} │ {s.idle_cycles:7d} │ {s.total_ops:9d} │"
            )
    
    lines.extend([
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│                    BOTTLENECK ANALYSIS                            │",
        "├" + "─" * 68 + "┤",
        f"│ Primary Bottleneck:        {analysis.primary_bottleneck:10s}                          │",
        f"│ Secondary Bottleneck:      {analysis.secondary_bottleneck:10s}                          │",
        f"│ Concurrent Saturation:     {analysis.concurrent_saturation:5d} cycles                        │",
        f"│ Load/VALU Competition:     {analysis.load_valu_competition:5d} cycles                        │",
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│                    PACKING EFFICIENCY                             │",
        "├" + "─" * 68 + "┤",
        f"│ Avg Slots Per Cycle:       {analysis.avg_slots_per_cycle:5.2f} / {analysis.max_possible_slots}                          │",
        f"│ Packing Efficiency:        {analysis.packing_efficiency:5.1f}%                              │",
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│                    PHASE BREAKDOWN                                │",
        "├" + "─" * 68 + "┤",
        f"│ Init Phase:                {analysis.init_phase_cycles:5d} cycles                        │",
        f"│ Compute Phase:             {analysis.compute_phase_cycles:5d} cycles                        │",
        f"│ Store Phase:               {analysis.store_phase_cycles:5d} cycles                        │",
        "└" + "─" * 68 + "┘",
        ""
    ])
    
    if analysis.recommendations:
        lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(analysis.recommendations, 1):
            lines.append(f"  {i}. {rec}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with actual kernel
    from perf_takehome import KernelBuilder
    
    kb = KernelBuilder()
    kb.build_kernel(10, 1023, 256, 16)
    
    analysis = analyze_instruction_entropy(kb.instrs)
    print(format_instruction_entropy(analysis))
