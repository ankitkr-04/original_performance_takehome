#!/usr/bin/env python3
"""
Critical Path MCP - Analyzes instruction dependencies and pipeline hazards
==========================================================================
Identifies Read-After-Write (RAW) dependencies, pipeline bubbles, and 
opportunities for better instruction interleaving.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from problem import SLOT_LIMITS, VLEN


@dataclass
class RAWHazard:
    """A Read-After-Write hazard"""
    cycle: int
    writer_engine: str
    reader_engine: str
    register: int
    bubble_cycles: int
    description: str


@dataclass
class DependencyChain:
    """A chain of dependent operations"""
    start_cycle: int
    end_cycle: int
    length: int
    operations: List[Tuple[int, str, str]]  # (cycle, engine, op_type)
    critical: bool  # Is this on the critical path?


@dataclass
class CriticalPathAnalysis:
    """Complete critical path analysis"""
    total_cycles: int
    raw_hazards: List[RAWHazard]
    total_bubble_cycles: int
    
    # Dependency patterns
    alu_to_valu_hazards: int
    valu_to_valu_hazards: int
    load_to_valu_hazards: int
    
    # Longest dependency chains
    longest_chains: List[DependencyChain]
    
    # Interleaving opportunities
    interleave_opportunities: List[str]
    
    # Bottleneck analysis
    load_bottleneck_cycles: int
    valu_bottleneck_cycles: int
    
    recommendations: List[str]


def analyze_critical_path(instrs: List[dict]) -> CriticalPathAnalysis:
    """
    Analyze instruction stream for critical path and dependencies.
    
    Architecture notes:
    - Zero-cycle bypass within same instruction
    - Results available at START of next cycle (1-cycle latency)
    - RAW hazard if reading a register written in immediately previous cycle
    """
    
    raw_hazards = []
    
    # Track what each instruction writes and reads
    # Register format: just the scratch address
    
    prev_writes = {}  # register -> (engine, cycle)
    
    alu_to_valu = 0
    valu_to_valu = 0
    load_to_valu = 0
    
    load_bottleneck = 0
    valu_bottleneck = 0
    
    for cycle, instr in enumerate(instrs):
        current_reads = defaultdict(list)  # register -> [engines reading]
        current_writes = {}  # register -> engine
        
        # Extract reads and writes from each engine
        for engine, slots in instr.items():
            if engine == 'debug':
                continue
                
            for slot in slots:
                if not slot:
                    continue
                    
                op = slot[0]
                
                if engine == 'load':
                    if op == 'const':
                        # const writes dst
                        dst = slot[1]
                        current_writes[dst] = engine
                    elif op == 'load':
                        # load dst, addr_reg
                        dst, addr = slot[1], slot[2]
                        current_reads[addr].append(engine)
                        current_writes[dst] = engine
                    elif op == 'vload':
                        # vload dst, addr_reg (dst is base of VLEN words)
                        dst, addr = slot[1], slot[2]
                        current_reads[addr].append(engine)
                        for i in range(VLEN):
                            current_writes[dst + i] = engine
                            
                elif engine == 'store':
                    if op == 'store':
                        addr, src = slot[1], slot[2]
                        current_reads[addr].append(engine)
                        current_reads[src].append(engine)
                    elif op == 'vstore':
                        addr, src = slot[1], slot[2]
                        current_reads[addr].append(engine)
                        for i in range(VLEN):
                            current_reads[src + i].append(engine)
                            
                elif engine in ('alu', 'valu'):
                    if op == 'vbroadcast':
                        dst, src = slot[1], slot[2]
                        current_reads[src].append(engine)
                        for i in range(VLEN):
                            current_writes[dst + i] = engine
                    elif op == 'multiply_add':
                        dst, a, b, c = slot[1], slot[2], slot[3], slot[4]
                        current_reads[a].append(engine)
                        current_reads[b].append(engine)
                        current_reads[c].append(engine)
                        if engine == 'valu':
                            for i in range(VLEN):
                                current_writes[dst + i] = engine
                        else:
                            current_writes[dst] = engine
                    elif len(slot) >= 4:
                        # Binary op: op, dst, src1, src2
                        dst, src1, src2 = slot[1], slot[2], slot[3]
                        current_reads[src1].append(engine)
                        current_reads[src2].append(engine)
                        if engine == 'valu':
                            for i in range(VLEN):
                                current_writes[dst + i] = engine
                        else:
                            current_writes[dst] = engine
        
        # Check for RAW hazards with previous instruction
        for reg, readers in current_reads.items():
            if reg in prev_writes:
                writer_engine, write_cycle = prev_writes[reg]
                if write_cycle == cycle - 1:
                    # Potential RAW hazard (1-cycle latency)
                    for reader in readers:
                        hazard = RAWHazard(
                            cycle=cycle,
                            writer_engine=writer_engine,
                            reader_engine=reader,
                            register=reg,
                            bubble_cycles=0,  # Architecture handles this
                            description=f"Cycle {cycle}: {reader} reads reg {reg} written by {writer_engine} at cycle {cycle-1}"
                        )
                        raw_hazards.append(hazard)
                        
                        if writer_engine == 'alu' and reader == 'valu':
                            alu_to_valu += 1
                        elif writer_engine == 'valu' and reader == 'valu':
                            valu_to_valu += 1
                        elif writer_engine == 'load' and reader == 'valu':
                            load_to_valu += 1
        
        # Check for bottlenecks (wanted more slots than available)
        if 'load' in instr and len(instr['load']) >= 2:
            # Could be a load bottleneck if we have lots of loads queued
            load_bottleneck += 1
        if 'valu' in instr and len(instr['valu']) >= 6:
            valu_bottleneck += 1
        
        # Update prev_writes
        for reg, engine in current_writes.items():
            prev_writes[reg] = (engine, cycle)
    
    # Generate recommendations
    recommendations = []
    
    if alu_to_valu > 10:
        recommendations.append(
            f"Found {alu_to_valu} ALU→VALU dependencies. "
            "Consider computing scalar values earlier or using VALU for all computations."
        )
    
    if load_to_valu > 50:
        recommendations.append(
            f"Found {load_to_valu} Load→VALU dependencies. "
            "This is expected for gather operations. Ensure proper pipelining overlap."
        )
    
    if load_bottleneck > len(instrs) * 0.3:
        recommendations.append(
            f"Load engine saturated in {load_bottleneck} cycles ({100*load_bottleneck/len(instrs):.1f}%). "
            "Consider precomputing addresses or muxing instead of loading."
        )
    
    if valu_bottleneck > len(instrs) * 0.5:
        recommendations.append(
            f"VALU engine saturated in {valu_bottleneck} cycles ({100*valu_bottleneck/len(instrs):.1f}%). "
            "Good utilization! Consider if any work can move to ALU."
        )
    
    interleave_opportunities = []
    
    # Find cycles with load-only or valu-only
    for i, instr in enumerate(instrs):
        has_load = 'load' in instr and instr['load']
        has_valu = 'valu' in instr and instr['valu']
        
        if has_load and not has_valu and i > 0 and i < len(instrs) - 1:
            # Load-only instruction - could potentially interleave
            if i < 10 or i > len(instrs) - 10:
                continue  # Skip init/cleanup
            if len(interleave_opportunities) < 5:
                interleave_opportunities.append(
                    f"Cycle {i}: Load-only instruction - consider interleaving VALU work here"
                )
    
    return CriticalPathAnalysis(
        total_cycles=len(instrs),
        raw_hazards=raw_hazards[:20],  # Limit output
        total_bubble_cycles=0,  # Architecture handles bypass
        alu_to_valu_hazards=alu_to_valu,
        valu_to_valu_hazards=valu_to_valu,
        load_to_valu_hazards=load_to_valu,
        longest_chains=[],  # TODO: implement chain detection
        interleave_opportunities=interleave_opportunities,
        load_bottleneck_cycles=load_bottleneck,
        valu_bottleneck_cycles=valu_bottleneck,
        recommendations=recommendations
    )


def format_critical_path(analysis: CriticalPathAnalysis) -> str:
    """Format critical path analysis as readable report."""
    lines = [
        "=" * 70,
        "CRITICAL PATH & DEPENDENCY ANALYSIS",
        "=" * 70,
        "",
        f"Total Instructions: {analysis.total_cycles}",
        "",
        "┌" + "─" * 68 + "┐",
        "│                    DEPENDENCY HAZARDS                             │",
        "├" + "─" * 68 + "┤",
        f"│ ALU → VALU hazards:     {analysis.alu_to_valu_hazards:5d}                                  │",
        f"│ VALU → VALU hazards:    {analysis.valu_to_valu_hazards:5d}                                  │",
        f"│ Load → VALU hazards:    {analysis.load_to_valu_hazards:5d}                                  │",
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│                    ENGINE SATURATION                              │",
        "├" + "─" * 68 + "┤",
        f"│ Load saturated cycles:  {analysis.load_bottleneck_cycles:5d} ({100*analysis.load_bottleneck_cycles/analysis.total_cycles:.1f}%)                        │",
        f"│ VALU saturated cycles:  {analysis.valu_bottleneck_cycles:5d} ({100*analysis.valu_bottleneck_cycles/analysis.total_cycles:.1f}%)                        │",
        "└" + "─" * 68 + "┘",
        ""
    ]
    
    if analysis.interleave_opportunities:
        lines.append("INTERLEAVING OPPORTUNITIES:")
        for opp in analysis.interleave_opportunities[:5]:
            lines.append(f"  • {opp}")
        lines.append("")
    
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
    
    analysis = analyze_critical_path(kb.instrs)
    print(format_critical_path(analysis))
