#!/usr/bin/env python3
"""
Tier 2 MCP: Gather Pressure Analyzer
======================================
Count scalar loads for node values, repeated loads, address recomputes.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_gather_pressure(instrs: List[Dict], verbose: bool = True) -> Dict:
    """
    Analyze memory access patterns and gather pressure.
    
    Args:
        instrs: List of instruction dictionaries
        verbose: Print detailed analysis
    
    Returns:
        Dictionary with gather statistics
    """
    scalar_loads = []
    vector_loads = []
    load_addresses = []
    address_computations = []
    
    # Track address register usage
    addr_defines = {}  # addr -> first definition cycle
    addr_uses = defaultdict(list)  # addr -> list of use cycles
    
    for i, instr in enumerate(instrs):
        # Track ALU operations that compute addresses
        for op in instr.get('alu', []):
            if len(op) >= 4 and op[0] in ['+', '*', '<<', '>>']:
                # Destination might be an address
                dest = op[1]
                address_computations.append((i, op))
                addr_defines[dest] = i
        
        # Track load operations
        for op in instr.get('load', []):
            if op[0] == 'load':
                scalar_loads.append((i, op))
                if len(op) >= 3:
                    addr = op[2]
                    load_addresses.append(addr)
                    if addr in addr_defines:
                        addr_uses[addr].append(i)
            elif op[0] == 'vload':
                vector_loads.append((i, op))
                if len(op) >= 3:
                    addr = op[2]
                    load_addresses.append(addr)
                    if addr in addr_defines:
                        addr_uses[addr].append(i)
    
    # Count repeated loads from same addresses
    addr_counts = Counter(load_addresses)
    repeated_addrs = {addr: count for addr, count in addr_counts.items() if count > 1}
    
    # Estimate gather operations (scalar loads that look like node value fetches)
    # These are loads that happen in sequence (gathering scattered data)
    gather_ops = 0
    gather_sequences = []
    current_seq = []
    
    for i, (cycle, op) in enumerate(scalar_loads):
        if op[0] == 'load':
            if current_seq and cycle == current_seq[-1][0] + 1:
                # Consecutive loads likely a gather sequence
                current_seq.append((cycle, op))
            else:
                if len(current_seq) > 3:
                    gather_sequences.append(current_seq)
                    gather_ops += len(current_seq)
                current_seq = [(cycle, op)]
    
    if len(current_seq) > 3:
        gather_sequences.append(current_seq)
        gather_ops += len(current_seq)
    
    # Analyze address lifetime (compute to use)
    addr_lifetimes = []
    for addr, uses in addr_uses.items():
        if addr in addr_defines:
            define_cycle = addr_defines[addr]
            for use_cycle in uses:
                lifetime = use_cycle - define_cycle
                addr_lifetimes.append((addr, define_cycle, use_cycle, lifetime))
    
    # Find address recomputes (same addr computed multiple times)
    addr_recomputes = defaultdict(list)
    for cycle, op in address_computations:
        # Simple pattern matching for address computation
        if len(op) >= 4:
            pattern = f"{op[0]}_{op[2]}_{op[3]}"  # op_src1_src2
            addr_recomputes[pattern].append(cycle)
    
    repeated_computes = {pattern: cycles for pattern, cycles in addr_recomputes.items() if len(cycles) > 1}
    
    if verbose:
        print(f"\n{'='*70}")
        print("GATHER PRESSURE ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'LOAD STATISTICS':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Scalar loads:':<40} {len(scalar_loads):>26} │")
        print(f"│ {'Vector loads:':<40} {len(vector_loads):>26} │")
        print(f"│ {'Estimated gather operations:':<40} {gather_ops:>26} │")
        print(f"│ {'Gather sequences:':<40} {len(gather_sequences):>26} │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'ADDRESS REUSE':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Unique addresses loaded:':<40} {len(addr_counts):>26} │")
        print(f"│ {'Addresses loaded multiple times:':<40} {len(repeated_addrs):>26} │")
        print(f"└{'─'*68}┘")
        
        if repeated_addrs:
            print(f"\n┌{'─'*68}┐")
            print(f"│ {'TOP REPEATED ADDRESS LOADS':^66} │")
            print(f"├{'─'*68}┤")
            print(f"│ {'Address':<30} │ {'Load Count':>33} │")
            print(f"├{'─'*68}┤")
            
            top_repeated = sorted(repeated_addrs.items(), key=lambda x: x[1], reverse=True)[:10]
            for addr, count in top_repeated:
                addr_str = str(addr) if not isinstance(addr, int) else f"Addr {addr}"
                print(f"│ {addr_str:<30} │ {count:>33} │")
            print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'ADDRESS COMPUTATION':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Total address computations:':<40} {len(address_computations):>26} │")
        print(f"│ {'Repeated computation patterns:':<40} {len(repeated_computes):>26} │")
        print(f"└{'─'*68}┘")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'ADDRESS LIFETIME ANALYSIS':^66} │")
        print(f"├{'─'*68}┤")
        
        if addr_lifetimes:
            avg_lifetime = sum(lt[3] for lt in addr_lifetimes) / len(addr_lifetimes)
            max_lifetime = max(addr_lifetimes, key=lambda x: x[3])
            long_lifetimes = [lt for lt in addr_lifetimes if lt[3] > 50]
            
            print(f"│ {'Average address lifetime:':<40} {avg_lifetime:>20.1f} cycles │")
            print(f"│ {'Max address lifetime:':<40} {max_lifetime[3]:>26} │")
            print(f"│ {'Addresses with >50 cycle gap:':<40} {len(long_lifetimes):>26} │")
            
            if long_lifetimes:
                print(f"│ {'':<66} │")
                print(f"│ {'Long-lived addresses (may waste registers):':<66} │")
                for addr, define, use, lifetime in sorted(long_lifetimes, key=lambda x: x[3], reverse=True)[:5]:
                    print(f"│   Addr {str(addr):<10} defined@{define:<6} used@{use:<6} gap={lifetime:<6}{' '*10}│")
        else:
            print(f"│ {'No address lifetime data available':<66} │")
        
        print(f"└{'─'*68}┘")
        
        # Recommendations
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'RECOMMENDATIONS':^66} │")
        print(f"├{'─'*68}┤")
        
        if gather_ops > 100:
            print(f"│ ⚠ High gather pressure ({gather_ops} ops)                             │")
            print(f"│   Consider prefetching or better data locality                    │")
        
        if len(repeated_addrs) > 20:
            print(f"│ ⚠ Many addresses loaded multiple times ({len(repeated_addrs)})                 │")
            print(f"│   Consider caching or reordering operations                       │")
        
        if len(repeated_computes) > 10:
            print(f"│ ⚠ Address computations repeated ({len(repeated_computes)} patterns)               │")
            print(f"│   Compute once and reuse address registers                        │")
        
        if addr_lifetimes and avg_lifetime > 30:
            print(f"│ ⚠ Long average address lifetime ({avg_lifetime:.0f} cycles)                   │")
            print(f"│   Compute addresses closer to use time                            │")
        
        print(f"└{'─'*68}┘")
    
    return {
        'scalar_loads': len(scalar_loads),
        'vector_loads': len(vector_loads),
        'gather_ops': gather_ops,
        'unique_addresses': len(addr_counts),
        'repeated_addresses': len(repeated_addrs),
        'addr_computations': len(address_computations),
        'repeated_computes': len(repeated_computes),
        'avg_addr_lifetime': sum(lt[3] for lt in addr_lifetimes) / len(addr_lifetimes) if addr_lifetimes else 0
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gather Pressure MCP - Analyze memory access patterns")
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--height', type=int, default=10)
    
    args = parser.parse_args()
    
    # Load kernel
    from perf_takehome import KernelBuilder
    from problem import Tree
    
    kb = KernelBuilder()
    
    import random
    random.seed(123)
    forest = Tree.generate(args.height)
    
    kb.build_kernel(forest.height, len(forest.values), args.batch, args.rounds)
    
    analyze_gather_pressure(kb.instrs, verbose=True)


if __name__ == "__main__":
    main()
