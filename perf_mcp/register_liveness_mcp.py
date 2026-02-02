"""
Register liveness analysis - track which vector registers are "live" at each point.
Helps identify register pressure issues and reuse opportunities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from problem import Tree, VLEN


def analyze_register_liveness(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Analyze vector register usage and liveness.
    
    Tracks:
    - When each register is first written
    - When each register is last read
    - Peak live register count
    - Register reuse opportunities
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
    
    # Track register writes and reads
    # A "register" here is a scratch memory location
    writes = {}  # addr -> first write instruction
    reads = {}   # addr -> last read instruction
    
    for i, instr in enumerate(instrs):
        for engine in ['load', 'valu', 'alu', 'store']:
            if engine not in instr:
                continue
            
            for op in instr[engine]:
                if engine == 'load':
                    if op[0] in ('const', 'load', 'vload', 'vbroadcast'):
                        # Destination is op[1]
                        dest = op[1]
                        if dest not in writes:
                            writes[dest] = i
                        # Source (if any) is op[2]
                        if len(op) > 2 and isinstance(op[2], int):
                            reads[op[2]] = i
                elif engine == 'valu':
                    # Format: (op, dest, src1, src2, ...)
                    if len(op) >= 2:
                        dest = op[1]
                        if dest not in writes:
                            writes[dest] = i
                        # Sources
                        for src in op[2:]:
                            if isinstance(src, int):
                                reads[src] = i
                elif engine == 'alu':
                    # Format: (op, dest, src1, src2)
                    if len(op) >= 2:
                        dest = op[1]
                        if dest not in writes:
                            writes[dest] = i
                        for src in op[2:]:
                            if isinstance(src, int):
                                reads[src] = i
                elif engine == 'store':
                    # vstore: (op, addr, value)
                    if len(op) >= 3:
                        reads[op[1]] = i  # Address
                        reads[op[2]] = i  # Value
    
    # Calculate liveness per instruction
    liveness = []
    for i in range(len(instrs)):
        live_count = 0
        for addr in writes:
            write_time = writes[addr]
            read_time = reads.get(addr, write_time)
            if write_time <= i <= read_time:
                live_count += 1
        liveness.append(live_count)
    
    peak_liveness = max(liveness) if liveness else 0
    peak_idx = liveness.index(peak_liveness) if liveness else 0
    
    # Find registers with long lifetimes
    long_lived = []
    for addr in writes:
        write_time = writes[addr]
        read_time = reads.get(addr, write_time)
        lifetime = read_time - write_time
        if lifetime > 100:
            long_lived.append({
                "addr": addr,
                "write": write_time,
                "read": read_time,
                "lifetime": lifetime,
            })
    
    long_lived.sort(key=lambda x: -x['lifetime'])
    
    # Calculate scratch layout from debug info
    scratch_info = []
    for addr, (name, length) in kb.scratch_debug.items():
        write_time = writes.get(addr, -1)
        read_time = reads.get(addr, -1)
        scratch_info.append({
            "name": name,
            "addr": addr,
            "length": length,
            "first_write": write_time,
            "last_read": read_time,
        })
    
    return {
        "total_registers": len(writes),
        "peak_liveness": peak_liveness,
        "peak_at_instruction": peak_idx,
        "scratch_used": kb.scratch_ptr,
        "long_lived_registers": long_lived[:10],
        "scratch_layout": scratch_info[:20],
        "liveness_histogram": {
            "0-100": sum(1 for l in liveness if l < 100),
            "100-200": sum(1 for l in liveness if 100 <= l < 200),
            "200-300": sum(1 for l in liveness if 200 <= l < 300),
            "300+": sum(1 for l in liveness if l >= 300),
        },
    }


if __name__ == "__main__":
    print("=== Register Liveness: perf_takehome ===")
    result = analyze_register_liveness("perf_takehome")
    print(f"Total registers: {result['total_registers']}")
    print(f"Peak liveness: {result['peak_liveness']} at instruction {result['peak_at_instruction']}")
    print(f"Scratch used: {result['scratch_used']}")
    print(f"Liveness histogram: {result['liveness_histogram']}")
    print(f"\nLong-lived registers (top 5):")
    for r in result['long_lived_registers'][:5]:
        print(f"  addr {r['addr']}: lifetime {r['lifetime']} (write@{r['write']}, read@{r['read']})")
    
    print("\n=== Register Liveness: takehome_diff ===")
    result = analyze_register_liveness("takehome_diff")
    print(f"Total registers: {result['total_registers']}")
    print(f"Peak liveness: {result['peak_liveness']} at instruction {result['peak_at_instruction']}")
    print(f"Scratch used: {result['scratch_used']}")
