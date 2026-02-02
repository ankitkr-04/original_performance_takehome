"""
DependencyLatencyTracer - Track producer-consumer latency for bypass optimization.

The simulator has 1-cycle latency: results of Cycle N are available for Cycle N+1.
This tool finds where address calculation and loads are too far apart (wasting buffers)
or identifies true dependencies that prevent further optimization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, Machine, VLEN, SLOT_LIMITS


def analyze_dependency_latency(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Trace producer-consumer relationships and measure latency between them.
    
    Returns:
    - Address-to-load latency (how soon after addr calc do we use it)
    - Back-to-back patterns (minimum latency = 1 cycle, perfect)
    - "Too safe" patterns (latency > 5 cycles, wasting scratch)
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
    
    # Track when each address is produced (written)
    production_cycle = {}  # address -> cycle it was last written
    
    # Track consumer latencies
    latencies = []  # List of (address, producer_cycle, consumer_cycle, latency)
    
    # Track STALE READ HAZARDS (write and read in same cycle = read sees OLD value)
    stale_read_hazards = []
    
    for cycle, instr in enumerate(instrs):
        # Find all writes in this instruction (producers)
        writes_this_cycle = set()
        
        for alu_op in instr.get('alu', []):
            if len(alu_op) >= 2:
                dest = alu_op[1]
                writes_this_cycle.add(dest)
        
        for valu_op in instr.get('valu', []):
            if len(valu_op) >= 2:
                dest = valu_op[1]
                if valu_op[0] == 'vbroadcast':
                    for i in range(VLEN):
                        writes_this_cycle.add(dest + i)
                else:
                    for i in range(VLEN):
                        writes_this_cycle.add(dest + i)
        
        for load_op in instr.get('load', []):
            if load_op[0] == 'const' and len(load_op) >= 2:
                writes_this_cycle.add(load_op[1])
            elif load_op[0] in ['load', 'load_offset'] and len(load_op) >= 2:
                writes_this_cycle.add(load_op[1])
            elif load_op[0] == 'vload' and len(load_op) >= 2:
                for i in range(VLEN):
                    writes_this_cycle.add(load_op[1] + i)
        
        # Find all reads in this instruction (consumers)
        reads_this_cycle = set()
        
        for alu_op in instr.get('alu', []):
            if len(alu_op) >= 4:
                reads_this_cycle.add(alu_op[2])
                reads_this_cycle.add(alu_op[3])
        
        for valu_op in instr.get('valu', []):
            if len(valu_op) >= 3:
                for src_base in valu_op[2:]:
                    if isinstance(src_base, int):
                        for i in range(VLEN):
                            reads_this_cycle.add(src_base + i)
        
        for load_op in instr.get('load', []):
            if load_op[0] == 'load' and len(load_op) >= 3:
                reads_this_cycle.add(load_op[2])  # Address source
            elif load_op[0] == 'load_offset' and len(load_op) >= 3:
                reads_this_cycle.add(load_op[2])  # Address source
        
        for store_op in instr.get('store', []):
            if len(store_op) >= 3:
                reads_this_cycle.add(store_op[1])  # Address
                reads_this_cycle.add(store_op[2])  # Value
        
        # Calculate latencies for reads
        for addr in reads_this_cycle:
            if addr in production_cycle:
                latency = cycle - production_cycle[addr]
                if latency == 0:
                    # STALE READ HAZARD: Write and read in same instruction!
                    # The read will see the OLD value from before this instruction.
                    stale_read_hazards.append({
                        'cycle': cycle,
                        'address': addr,
                        'description': f"Cycle {cycle}: Read-After-Write in same instruction. "
                                      f"Reader sees OLD value of scratch[{addr}].",
                    })
                elif latency >= 1:  # Valid (result available next cycle)
                    latencies.append({
                        'address': addr,
                        'producer_cycle': production_cycle[addr],
                        'consumer_cycle': cycle,
                        'latency': latency,
                    })
        
        # Check for same-cycle RAW hazards (write and read in same instruction)
        same_cycle_raw = writes_this_cycle & reads_this_cycle
        for addr in same_cycle_raw:
            if addr not in [h['address'] for h in stale_read_hazards if h['cycle'] == cycle]:
                stale_read_hazards.append({
                    'cycle': cycle,
                    'address': addr,
                    'description': f"Cycle {cycle}: Same-instruction RAW on scratch[{addr}]. "
                                  f"Common bug: v_addr calc + load in same VLIW word.",
                })
        
        # Update production times
        for addr in writes_this_cycle:
            production_cycle[addr] = cycle
    
    # Analyze latency distribution
    if not latencies:
        return {"error": "No producer-consumer pairs found"}
    
    latency_values = [l['latency'] for l in latencies]
    
    # Categorize
    back_to_back = [l for l in latencies if l['latency'] == 1]  # Perfect
    short_latency = [l for l in latencies if 2 <= l['latency'] <= 5]  # Good
    medium_latency = [l for l in latencies if 6 <= l['latency'] <= 20]  # Could improve
    long_latency = [l for l in latencies if l['latency'] > 20]  # Too safe / wasting scratch
    
    # Find critical paths (chains of dependent operations)
    # Simplified: look for addresses that are both read and written frequently
    address_frequency = {}
    for l in latencies:
        addr = l['address']
        if addr not in address_frequency:
            address_frequency[addr] = {'reads': 0, 'avg_latency': 0, 'latencies': []}
        address_frequency[addr]['reads'] += 1
        address_frequency[addr]['latencies'].append(l['latency'])
    
    for addr, freq in address_frequency.items():
        freq['avg_latency'] = sum(freq['latencies']) / len(freq['latencies'])
    
    # Hot addresses (frequently accessed with various latencies)
    hot_addresses = sorted(
        [(addr, data) for addr, data in address_frequency.items() if data['reads'] > 10],
        key=lambda x: -x[1]['reads']
    )[:20]
    
    avg_latency = sum(latency_values) / len(latency_values)
    min_latency = min(latency_values)
    max_latency = max(latency_values)
    
    return {
        "total_dependencies": len(latencies),
        "latency_stats": {
            "min": min_latency,
            "max": max_latency,
            "avg": round(avg_latency, 2),
            "median": sorted(latency_values)[len(latency_values) // 2],
        },
        "latency_distribution": {
            "back_to_back_1": len(back_to_back),
            "short_2_5": len(short_latency),
            "medium_6_20": len(medium_latency),
            "long_20plus": len(long_latency),
        },
        "back_to_back_pct": round(len(back_to_back) / len(latencies) * 100, 1),
        "wasted_buffer_potential": len(long_latency),
        "hot_addresses": [(addr, data['reads'], round(data['avg_latency'], 1)) 
                        for addr, data in hot_addresses[:10]],
        "sample_long_latencies": [
            {
                'address': l['address'],
                'producer': l['producer_cycle'],
                'consumer': l['consumer_cycle'],
                'latency': l['latency'],
            }
            for l in sorted(long_latency, key=lambda x: -x['latency'])[:10]
        ],
        # STALE READ HAZARDS - critical bugs!
        "stale_read_hazards": stale_read_hazards[:20],
        "stale_read_count": len(stale_read_hazards),
        "has_stale_read_bugs": len(stale_read_hazards) > 0,
        "optimization_hints": [
            f"CRITICAL: {len(stale_read_hazards)} stale read hazards detected!" if stale_read_hazards else "No stale read hazards (good)",
            f"{len(back_to_back)} dependencies at minimum latency (1 cycle) - optimal",
            f"{len(long_latency)} dependencies with >20 cycle latency - review for scratch reuse",
            f"Average latency {avg_latency:.1f} cycles - target is 1-5 for tight pipelining",
        ],
    }


def check_address_load_timing(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Specifically check the timing between address calculation and memory loads.
    
    For scatter-gather, we need:
    1. Calculate address (ALU: base + offset)
    2. Issue load (Load: mem[addr])
    
    Optimal: address ready at cycle N, load issued at cycle N+1
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
    
    # Track address production and load consumption
    addr_calc_cycle = {}  # scratch address -> cycle it was computed
    load_timings = []  # List of (addr_scratch_loc, addr_calc_cycle, load_cycle, gap)
    
    for cycle, instr in enumerate(instrs):
        # Find address calculations (typically ALU ops that compute into v_addr range)
        for alu_op in instr.get('alu', []):
            if len(alu_op) >= 4 and alu_op[0] == '+':
                dest = alu_op[1]
                addr_calc_cycle[dest] = cycle
        
        # Also track VALU ops that might compute addresses
        for valu_op in instr.get('valu', []):
            if len(valu_op) >= 2 and valu_op[0] == '+':
                dest = valu_op[1]
                for i in range(VLEN):
                    addr_calc_cycle[dest + i] = cycle
        
        # Find loads that use addresses
        for load_op in instr.get('load', []):
            if load_op[0] in ['load', 'load_offset'] and len(load_op) >= 3:
                addr_scratch = load_op[2]  # Scratch location holding the address
                if addr_scratch in addr_calc_cycle:
                    gap = cycle - addr_calc_cycle[addr_scratch]
                    load_timings.append({
                        'addr_scratch': addr_scratch,
                        'calc_cycle': addr_calc_cycle[addr_scratch],
                        'load_cycle': cycle,
                        'gap': gap,
                    })
    
    if not load_timings:
        return {"note": "No address->load patterns detected"}
    
    gaps = [t['gap'] for t in load_timings]
    
    # Categorize
    optimal = [t for t in load_timings if t['gap'] == 1]  # Perfect timing
    good = [t for t in load_timings if 2 <= t['gap'] <= 5]
    wasteful = [t for t in load_timings if t['gap'] > 10]
    
    return {
        "total_address_load_pairs": len(load_timings),
        "gap_stats": {
            "min": min(gaps),
            "max": max(gaps),
            "avg": round(sum(gaps) / len(gaps), 2),
        },
        "timing_quality": {
            "optimal_gap_1": len(optimal),
            "good_gap_2_5": len(good),
            "wasteful_gap_10plus": len(wasteful),
        },
        "optimal_pct": round(len(optimal) / len(load_timings) * 100, 1),
        "sample_wasteful": [
            f"Addr scratch {t['addr_scratch']}: calc@{t['calc_cycle']}, load@{t['load_cycle']}, gap={t['gap']}"
            for t in sorted(wasteful, key=lambda x: -x['gap'])[:5]
        ],
        "recommendation": "Good timing - addresses used promptly after calculation" 
                         if len(wasteful) < len(load_timings) * 0.1
                         else f"Consider tightening {len(wasteful)} address-load gaps to free scratch earlier",
    }


if __name__ == "__main__":
    print("=== Dependency Latency Analysis: perf_takehome ===")
    result = analyze_dependency_latency("perf_takehome")
    print(f"Total dependencies: {result['total_dependencies']}")
    print(f"\nLatency stats:")
    for k, v in result['latency_stats'].items():
        print(f"  {k}: {v}")
    print(f"\nLatency distribution:")
    for k, v in result['latency_distribution'].items():
        print(f"  {k}: {v}")
    print(f"\nBack-to-back percentage: {result['back_to_back_pct']}%")
    print(f"\nOptimization hints:")
    for hint in result['optimization_hints']:
        print(f"  - {hint}")
    
    print("\n=== Address-Load Timing Check ===")
    timing = check_address_load_timing("perf_takehome")
    print(f"Total address->load pairs: {timing.get('total_address_load_pairs', 0)}")
    if 'gap_stats' in timing:
        print(f"Gap stats: {timing['gap_stats']}")
        print(f"Timing quality: {timing['timing_quality']}")
        print(f"Optimal percentage: {timing['optimal_pct']}%")
        print(f"Recommendation: {timing['recommendation']}")
