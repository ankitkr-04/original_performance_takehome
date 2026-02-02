"""
ScratchLivenessHeatmap - Map scratch address usage over time.

The 1536 limit is the biggest hurdle for triple buffering.
This tool shows "Dead Zones" - scratch space allocated but unused for many cycles.
These can be reused (aliased) for different purposes at different times.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, Machine, VLEN, SLOT_LIMITS, SCRATCH_SIZE


def analyze_scratch_liveness(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Analyze scratch register liveness over time.
    
    Returns:
    - When each scratch address is written (born)
    - When each scratch address is last read (dies)
    - "Dead zones" where addresses are allocated but not used
    - Aliasing opportunities for scratch reuse
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
    
    # Track when each address is written and read
    # address -> {first_write, last_write, first_read, last_read, write_count, read_count}
    address_lifetime = {}
    
    # Track vector ranges to detect scalar overwrites ("hole punching")
    # Vector base address -> (cycle_written, set of addresses in vector)
    vector_ranges = {}  # base_addr -> {'cycle': cycle, 'addresses': set()}
    scalar_overwrites = []  # List of potential bugs: scalar writes into vector ranges
    
    for cycle, instr in enumerate(instrs):
        # Track SCALAR writes first (to detect overwrites into vectors)
        scalar_writes_this_cycle = set()
        
        for alu_op in instr.get('alu', []):
            # alu format: (op, dest, src1, src2) - SCALAR operation
            if len(alu_op) >= 2:
                dest = alu_op[1]
                scalar_writes_this_cycle.add(dest)
                if dest not in address_lifetime:
                    address_lifetime[dest] = {'first_write': cycle, 'last_write': cycle, 
                                              'first_read': None, 'last_read': None,
                                              'write_count': 0, 'read_count': 0,
                                              'is_vector': False}
                address_lifetime[dest]['last_write'] = cycle
                address_lifetime[dest]['write_count'] += 1
        
        for load_op in instr.get('load', []):
            if load_op[0] == 'const' and len(load_op) >= 2:
                dest = load_op[1]
                scalar_writes_this_cycle.add(dest)
            elif load_op[0] in ['load', 'load_offset'] and len(load_op) >= 2:
                dest = load_op[1]
                scalar_writes_this_cycle.add(dest)
        
        # Check if any scalar write "punches a hole" in a vector range
        for scalar_addr in scalar_writes_this_cycle:
            for base_addr, vec_info in vector_ranges.items():
                if scalar_addr in vec_info['addresses'] and scalar_addr != base_addr:
                    # Scalar write into middle of a vector range - likely a bug!
                    scalar_overwrites.append({
                        'cycle': cycle,
                        'scalar_addr': scalar_addr,
                        'vector_base': base_addr,
                        'vector_written_at': vec_info['cycle'],
                        'description': f"Cycle {cycle}: Scalar write to scratch[{scalar_addr}] "
                                      f"punches hole in vector range [{base_addr}:{base_addr+VLEN}] "
                                      f"(vector written at cycle {vec_info['cycle']})",
                    })
        
        # Track writes (destinations)
        for alu_op in instr.get('alu', []):
            # alu format: (op, dest, src1, src2)
            if len(alu_op) >= 2:
                dest = alu_op[1]
                if dest not in address_lifetime:
                    address_lifetime[dest] = {'first_write': cycle, 'last_write': cycle, 
                                              'first_read': None, 'last_read': None,
                                              'write_count': 0, 'read_count': 0}
                address_lifetime[dest]['last_write'] = cycle
                address_lifetime[dest]['write_count'] += 1
        
        for valu_op in instr.get('valu', []):
            # valu format: (op, dest, ...) for most ops
            if len(valu_op) >= 2 and valu_op[0] != 'vbroadcast':
                dest = valu_op[1]
                # Track this as a vector range
                vector_ranges[dest] = {
                    'cycle': cycle,
                    'addresses': set(range(dest, dest + VLEN)),
                }
                # For vector ops, dest is base address
                for i in range(VLEN):
                    addr = dest + i
                    if addr not in address_lifetime:
                        address_lifetime[addr] = {'first_write': cycle, 'last_write': cycle,
                                                  'first_read': None, 'last_read': None,
                                                  'write_count': 0, 'read_count': 0,
                                                  'is_vector': True, 'vector_base': dest}
                    address_lifetime[addr]['last_write'] = cycle
                    address_lifetime[addr]['write_count'] += 1
            elif valu_op[0] == 'vbroadcast' and len(valu_op) >= 2:
                dest = valu_op[1]
                # Track this as a vector range
                vector_ranges[dest] = {
                    'cycle': cycle,
                    'addresses': set(range(dest, dest + VLEN)),
                }
                for i in range(VLEN):
                    addr = dest + i
                    if addr not in address_lifetime:
                        address_lifetime[addr] = {'first_write': cycle, 'last_write': cycle,
                                                  'first_read': None, 'last_read': None,
                                                  'write_count': 0, 'read_count': 0,
                                                  'is_vector': True, 'vector_base': dest}
                    address_lifetime[addr]['last_write'] = cycle
                    address_lifetime[addr]['write_count'] += 1
        
        for load_op in instr.get('load', []):
            if load_op[0] == 'const' and len(load_op) >= 2:
                dest = load_op[1]
                if dest not in address_lifetime:
                    address_lifetime[dest] = {'first_write': cycle, 'last_write': cycle,
                                              'first_read': None, 'last_read': None,
                                              'write_count': 0, 'read_count': 0}
                address_lifetime[dest]['last_write'] = cycle
                address_lifetime[dest]['write_count'] += 1
            elif load_op[0] in ['load', 'load_offset'] and len(load_op) >= 2:
                dest = load_op[1]
                if dest not in address_lifetime:
                    address_lifetime[dest] = {'first_write': cycle, 'last_write': cycle,
                                              'first_read': None, 'last_read': None,
                                              'write_count': 0, 'read_count': 0}
                address_lifetime[dest]['last_write'] = cycle
                address_lifetime[dest]['write_count'] += 1
            elif load_op[0] == 'vload' and len(load_op) >= 2:
                dest = load_op[1]
                for i in range(VLEN):
                    addr = dest + i
                    if addr not in address_lifetime:
                        address_lifetime[addr] = {'first_write': cycle, 'last_write': cycle,
                                                  'first_read': None, 'last_read': None,
                                                  'write_count': 0, 'read_count': 0}
                    address_lifetime[addr]['last_write'] = cycle
                    address_lifetime[addr]['write_count'] += 1
        
        # Track reads (sources)
        for alu_op in instr.get('alu', []):
            if len(alu_op) >= 4:
                for src in [alu_op[2], alu_op[3]]:
                    if src in address_lifetime:
                        if address_lifetime[src]['first_read'] is None:
                            address_lifetime[src]['first_read'] = cycle
                        address_lifetime[src]['last_read'] = cycle
                        address_lifetime[src]['read_count'] += 1
        
        for valu_op in instr.get('valu', []):
            if len(valu_op) >= 3:
                # Sources start at index 2
                for src_base in valu_op[2:]:
                    if isinstance(src_base, int):
                        for i in range(VLEN):
                            addr = src_base + i
                            if addr in address_lifetime:
                                if address_lifetime[addr]['first_read'] is None:
                                    address_lifetime[addr]['first_read'] = cycle
                                address_lifetime[addr]['last_read'] = cycle
                                address_lifetime[addr]['read_count'] += 1
    
    # Calculate liveness spans
    max_addr_used = max(address_lifetime.keys()) if address_lifetime else 0
    total_cycles = len(instrs)
    
    # Find addresses with large "dead" periods
    dead_zones = []
    for addr, life in address_lifetime.items():
        if life['last_read'] is not None:
            # Time from last write to first use
            write_to_read_gap = life['first_read'] - life['first_write'] if life['first_read'] else 0
            # Time from last read to end
            read_to_end = total_cycles - life['last_read']
            
            if write_to_read_gap > 100:
                dead_zones.append({
                    'address': addr,
                    'type': 'early_allocation',
                    'gap': write_to_read_gap,
                    'detail': f'Written at {life["first_write"]}, not read until {life["first_read"]}',
                })
            
            if read_to_end > 200 and addr < 500:  # Early addresses unused at end
                dead_zones.append({
                    'address': addr,
                    'type': 'late_abandonment',
                    'gap': read_to_end,
                    'detail': f'Last read at {life["last_read"]}, kernel ends at {total_cycles}',
                })
    
    # Group addresses by usage pattern
    # "Constants" - written once, read many times
    constants = [addr for addr, life in address_lifetime.items() 
                 if life['write_count'] == 1 and life.get('read_count', 0) > 10]
    
    # "Temporaries" - written and read few times
    temporaries = [addr for addr, life in address_lifetime.items()
                   if life['write_count'] <= 2 and life.get('read_count', 0) <= 5]
    
    # "Hot" registers - frequently accessed
    hot_registers = [addr for addr, life in address_lifetime.items()
                     if life['write_count'] + life.get('read_count', 0) > 50]
    
    # Calculate pressure over time (how many addresses are "live" at each point)
    # Simplified: bucket into time windows
    window_size = total_cycles // 10
    pressure_timeline = []
    
    for window in range(10):
        start = window * window_size
        end = start + window_size
        
        live_count = 0
        for addr, life in address_lifetime.items():
            # Address is "live" if written before window and read after window start
            if life['first_write'] <= end:
                last_use = life['last_read'] if life['last_read'] else life['last_write']
                if last_use >= start:
                    live_count += 1
        
        pressure_timeline.append({
            'window': window,
            'cycle_range': f'{start}-{end}',
            'live_addresses': live_count,
            'pressure_pct': round(live_count / SCRATCH_SIZE * 100, 1),
        })
    
    peak_pressure = max(p['live_addresses'] for p in pressure_timeline)
    peak_window = next(p for p in pressure_timeline if p['live_addresses'] == peak_pressure)
    
    return {
        "total_addresses_used": len(address_lifetime),
        "max_address": max_addr_used,
        "scratch_limit": SCRATCH_SIZE,
        "utilization_pct": round(max_addr_used / SCRATCH_SIZE * 100, 1),
        "peak_live_addresses": peak_pressure,
        "peak_pressure_pct": round(peak_pressure / SCRATCH_SIZE * 100, 1),
        "peak_at_window": peak_window,
        "pressure_timeline": pressure_timeline,
        "dead_zones_count": len(dead_zones),
        "dead_zones_sample": sorted(dead_zones, key=lambda x: -x['gap'])[:10],
        "address_categories": {
            "constants": len(constants),
            "temporaries": len(temporaries),
            "hot_registers": len(hot_registers),
        },
        # SCALAR OVERWRITE DETECTION - potential bugs!
        "scalar_overwrites": scalar_overwrites[:20],
        "scalar_overwrite_count": len(scalar_overwrites),
        "has_scalar_overwrite_bugs": len(scalar_overwrites) > 0,
        "vector_ranges_tracked": len(vector_ranges),
        "aliasing_opportunities": [
            f"Early constants (cycle 0-100) can alias with late store buffers (cycle {total_cycles-200}+)",
            f"Found {len([d for d in dead_zones if d['type'] == 'early_allocation'])} early allocations with long gaps",
            f"Found {len([d for d in dead_zones if d['type'] == 'late_abandonment'])} addresses abandoned early",
        ] if dead_zones else ["No obvious aliasing opportunities found"],
        "warnings": [
            f"CRITICAL: {len(scalar_overwrites)} scalar writes into vector ranges detected!"
        ] if scalar_overwrites else [],
    }


def visualize_scratch_density(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Produce a summary of peak memory pressure points.
    """
    result = analyze_scratch_liveness(kernel_module, tree_height, rounds, batch_size)
    
    print(f"=== Scratch Memory Density: {kernel_module} ===")
    print(f"Total addresses used: {result['total_addresses_used']}")
    print(f"Max address: {result['max_address']} / {result['scratch_limit']}")
    print(f"Peak pressure: {result['peak_pressure_pct']}% ({result['peak_live_addresses']} live)")
    print(f"\nPressure Timeline:")
    
    for p in result['pressure_timeline']:
        bar_len = int(p['pressure_pct'] / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {p['cycle_range']:>12}: {bar} {p['pressure_pct']:5.1f}%")
    
    print(f"\nAliasing opportunities:")
    for opp in result['aliasing_opportunities']:
        print(f"  - {opp}")
    
    return result


if __name__ == "__main__":
    visualize_scratch_density("perf_takehome")
    print()
    
    print("=== Detailed Liveness Analysis ===")
    result = analyze_scratch_liveness("perf_takehome")
    print(f"Address categories: {result['address_categories']}")
    print(f"\nTop dead zones (longest gaps):")
    for dz in result['dead_zones_sample'][:5]:
        print(f"  Address {dz['address']}: {dz['type']}, gap={dz['gap']} cycles")
        print(f"    {dz['detail']}")
