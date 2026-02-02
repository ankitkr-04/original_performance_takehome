#!/usr/bin/env python3
"""
Tier 3 MCP: Address Lifetime Analyzer
=======================================
Track when v_addr produced, when consumed, flag over-buffering gaps.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class AddressLifetime:
    """Track lifetime of an address register"""
    addr_name: str
    addr_id: int
    define_instr: int
    use_instrs: List[int]
    lifetime: int  # Cycles from define to last use
    is_long_lived: bool  # > 50 cycles


def analyze_address_lifetimes(instrs: List[Dict], verbose: bool = True) -> Dict:
    """
    Analyze address register lifetimes and identify over-buffering.
    
    Args:
        instrs: List of instruction dictionaries
        verbose: Print detailed analysis
    
    Returns:
        Dictionary with lifetime statistics
    """
    # Track address definitions and uses
    addr_defs = {}  # addr -> instr_num
    addr_uses = defaultdict(list)  # addr -> [instr_nums]
    addr_names = {}  # addr -> name
    
    # Pattern 1: Address computation via ALU (forest_p + index)
    # Pattern 2: vbroadcast + add (v_addr = forest_p + indices)
    
    for i, instr in enumerate(instrs):
        # Track ALU operations that compute addresses
        for op in instr.get('alu', []):
            if len(op) >= 4 and op[0] == '+':
                dest = op[1]
                # Looks like address computation
                if 'addr' in str(dest) or 'base' in str(dest):
                    addr_defs[dest] = i
                    addr_names[dest] = f"addr_{dest}"
        
        # Track valu operations for address computation
        for op in instr.get('valu', []):
            if len(op) >= 2:
                if op[0] == 'vbroadcast':
                    dest = op[1]
                    if 'addr' in str(dest):
                        addr_defs[dest] = i
                        addr_names[dest] = f"v_addr_{dest}"
                elif op[0] == '+' and len(op) >= 4:
                    dest = op[1]
                    if 'addr' in str(dest):
                        addr_defs[dest] = i
                        addr_names[dest] = f"v_addr_{dest}"
        
        # Track load operations that use addresses
        for op in instr.get('load', []):
            if op[0] in ['load', 'vload'] and len(op) >= 3:
                addr = op[2]
                if addr in addr_defs:
                    addr_uses[addr].append(i)
    
    # Compute lifetimes
    lifetimes = []
    
    for addr, define_instr in addr_defs.items():
        uses = addr_uses.get(addr, [])
        if uses:
            last_use = max(uses)
            lifetime = last_use - define_instr
            is_long = lifetime > 50
            
            lifetimes.append(AddressLifetime(
                addr_name=addr_names.get(addr, str(addr)),
                addr_id=addr,
                define_instr=define_instr,
                use_instrs=uses,
                lifetime=lifetime,
                is_long_lived=is_long
            ))
    
    # Sort by lifetime
    lifetimes.sort(key=lambda x: x.lifetime, reverse=True)
    
    # Statistics
    if lifetimes:
        avg_lifetime = sum(lt.lifetime for lt in lifetimes) / len(lifetimes)
        long_lived = [lt for lt in lifetimes if lt.is_long_lived]
        short_lived = [lt for lt in lifetimes if lt.lifetime < 10]
    else:
        avg_lifetime = 0
        long_lived = []
        short_lived = []
    
    if verbose:
        print(f"\n{'='*70}")
        print("ADDRESS LIFETIME ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'SUMMARY STATISTICS':^66} │")
        print(f"├{'─'*68}┤")
        print(f"│ {'Total address registers:':<40} {len(lifetimes):>26} │")
        print(f"│ {'Average lifetime:':<40} {avg_lifetime:>20.1f} instrs │")
        print(f"│ {'Long-lived (>50 instrs):':<40} {len(long_lived):>26} │")
        print(f"│ {'Short-lived (<10 instrs):':<40} {len(short_lived):>26} │")
        print(f"└{'─'*68}┘")
        
        if long_lived:
            print(f"\n┌{'─'*68}┐")
            print(f"│ {'LONG-LIVED ADDRESSES (Potential Register Pressure)':^66} │")
            print(f"├{'─'*68}┤")
            print(f"│ {'Address':<20} │ {'Define@':>8} │ {'Last Use@':>10} │ {'Lifetime':>10} │")
            print(f"├{'─'*68}┤")
            
            for lt in long_lived[:15]:
                addr_str = lt.addr_name[:20]
                print(f"│ {addr_str:<20} │ {lt.define_instr:>8} │ {max(lt.use_instrs):>10} │ {lt.lifetime:>10} │")
            print(f"└{'─'*68}┘")
        
        # Show distribution
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'LIFETIME DISTRIBUTION':^66} │")
        print(f"├{'─'*68}┤")
        
        buckets = {
            '0-5': 0,
            '6-10': 0,
            '11-20': 0,
            '21-50': 0,
            '51-100': 0,
            '>100': 0
        }
        
        for lt in lifetimes:
            if lt.lifetime <= 5:
                buckets['0-5'] += 1
            elif lt.lifetime <= 10:
                buckets['6-10'] += 1
            elif lt.lifetime <= 20:
                buckets['11-20'] += 1
            elif lt.lifetime <= 50:
                buckets['21-50'] += 1
            elif lt.lifetime <= 100:
                buckets['51-100'] += 1
            else:
                buckets['>100'] += 1
        
        for bucket, count in buckets.items():
            pct = (count / len(lifetimes) * 100) if lifetimes else 0
            bar = '█' * int(pct / 2)
            print(f"│ {bucket:>8} instrs │ {count:>5} ({pct:>5.1f}%) {bar:<30} │")
        print(f"└{'─'*68}┘")
        
        # Detailed view of top offenders
        if long_lived:
            print(f"\n┌{'─'*68}┐")
            print(f"│ {'TOP 5 LONGEST-LIVED ADDRESSES (DETAILED)':^66} │")
            print(f"├{'─'*68}┤")
            
            for i, lt in enumerate(long_lived[:5]):
                print(f"│ #{i+1}: {lt.addr_name:<56} │")
                print(f"│   Defined at instr: {lt.define_instr:<47} │")
                print(f"│   Used at instrs:   {str(lt.use_instrs[:5])[1:-1]:<47} │")
                if len(lt.use_instrs) > 5:
                    print(f"│   ... ({len(lt.use_instrs)} total uses){' '*39} │")
                print(f"│   Lifetime: {lt.lifetime} instructions{' '*42} │")
                print(f"│   Register pressure: {'HIGH' if lt.lifetime > 100 else 'MEDIUM':<43} │")
                print(f"├{'─'*68}┤")
            print(f"└{'─'*68}┘")
        
        # Recommendations
        print(f"\n┌{'─'*68}┐")
        print(f"│ {'RECOMMENDATIONS':^66} │")
        print(f"├{'─'*68}┤")
        
        if len(long_lived) > 10:
            print(f"│ ⚠ Many long-lived addresses ({len(long_lived)} total)                     │")
            print(f"│   This creates register pressure. Consider:                       │")
            print(f"│   - Compute addresses closer to use time                          │")
            print(f"│   - Reuse address registers when possible                         │")
            print(f"│   - Use address generation closer to gather operations            │")
        
        if avg_lifetime > 40:
            print(f"│ ⚠ High average address lifetime ({avg_lifetime:.0f} instrs)                   │")
            print(f"│   Delay address computation until just before use                 │")
        
        if len(short_lived) > len(lifetimes) * 0.7:
            print(f"│ ✓ Most addresses are short-lived - good register usage            │")
        
        # Check for address reuse opportunities
        # Addresses with similar use patterns could share registers
        use_patterns = defaultdict(list)
        for lt in lifetimes:
            pattern = (len(lt.use_instrs), lt.lifetime // 10)
            use_patterns[pattern].append(lt)
        
        reusable = [addrs for addrs in use_patterns.values() if len(addrs) > 1]
        
        if reusable:
            print(f"│ {'':<66} │")
            print(f"│ 💡 Found {len(reusable)} groups of addresses with similar patterns          │")
            print(f"│   These might be candidates for register reuse                    │")
        
        print(f"└{'─'*68}┘")
    
    return {
        'total_addresses': len(lifetimes),
        'avg_lifetime': avg_lifetime,
        'long_lived': len(long_lived),
        'short_lived': len(short_lived),
        'lifetimes': lifetimes
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Address Lifetime MCP - Analyze address register usage")
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
    
    analyze_address_lifetimes(kb.instrs, verbose=True)


if __name__ == "__main__":
    main()
