"""
AlgebraicFusionAuditor - Find multiply_add fusion opportunities in hash implementation.

The "cheat code": The simulator's multiply_add can collapse multiple hash operations.
Hash stage pattern: a = ((a + c1) + (a << c2)) often becomes multiply_add(a, (1 << c2), c1)

This frees up VALU slots for muxing and address calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import HASH_STAGES


def analyze_hash_fusion_opportunities():
    """
    Analyze HASH_STAGES to find multiply_add fusion opportunities.
    
    Each stage has the pattern:
    1. a = a OP1 val1  (e.g., a = a + 12345)
    2. a = a OP2 a     (e.g., a = a ^ (a << 5))
    3. a = a OP3 val3  (e.g., a = a + 67890)
    
    Some patterns can be fused into multiply_add:
    - (a + c) + (a << n) = a * (1 + 2^n) + c (use multiply_add)
    - (a ^ c) ^ (a << n) = more complex, may not fuse
    """
    
    fusion_analysis = []
    total_ops_naive = 0
    total_ops_fused = 0
    
    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        stage_analysis = {
            "stage": i,
            "ops": f"({op1}, {val1}, {op2}, {op3}, {val3})",
            "naive_implementation": [],
            "fused_implementation": [],
            "savings": 0,
        }
        
        # Naive implementation (what most kernels do)
        # Step 1: a = a OP1 val1
        stage_analysis["naive_implementation"].append(f"valu {op1}: a = a {op1} {val1}")
        
        # Step 2: temp = a OP2 a (typically a << shift or a >> shift)
        # Then a = a ^ temp or a + temp
        if op2 in ["<<", ">>"]:
            stage_analysis["naive_implementation"].append(f"valu {op2}: temp = a {op2} (implicit shift)")
            stage_analysis["naive_implementation"].append(f"valu ^: a = a ^ temp")
        else:
            stage_analysis["naive_implementation"].append(f"valu {op2}: a = a {op2} a")
        
        # Step 3: a = a OP3 val3
        stage_analysis["naive_implementation"].append(f"valu {op3}: a = a {op3} {val3}")
        
        naive_ops = len(stage_analysis["naive_implementation"])
        total_ops_naive += naive_ops
        
        # Check for fusion opportunities
        can_fuse = False
        fusion_formula = None
        
        # Pattern 1: (a + c1) ^ (a << n) + c3
        # The shift+xor part is hard to fuse, but add operations can sometimes use multiply_add
        
        # Pattern 2: Look for (a + c) + (a << n) which = a * (2^n + 1) + c
        # This is the golden pattern for multiply_add
        
        # In practice, hash uses XOR not ADD for the middle operation
        # But the first and last operations are often ADD and can be combined
        
        if op1 == "+" and op3 == "+":
            # Can potentially use multiply_add for parts
            # a = a + val1; a = a ^ (a << shift); a = a + val3
            # After XOR: need to add val3, could have added (val1 + val3) at start
            # But XOR in middle breaks this
            
            # However, if we rearrange:
            # The shift+xor is: a' = (a + val1) ^ ((a + val1) << shift)
            # This expands to: a' = a + val1 + (a << shift) + (val1 << shift) with some XOR math
            # Complex, but some optimizations possible
            pass
        
        # The key insight: multiply_add(dest, a, b, c) = a*b + c
        # So if we have: result = a * constant + offset
        # We can do it in one instruction
        
        # For hash, the pattern a = a ^ (a << n) doesn't directly fuse
        # But we can optimize the surrounding adds
        
        # Simplified fusion: combine the two add operations
        if op1 == "+" and op3 == "+":
            fusion_formula = f"Combine adds: pre-add {val1}, post-add {val3}"
            can_fuse = True
        
        # Shift+XOR fusion: Use multiply_add for shift
        # a << n = a * (2^n), so (a << n) ^ a = not directly multiply_add
        # But broadcast the shift amount and use vector ops efficiently
        
        if can_fuse:
            stage_analysis["fused_implementation"] = [
                f"valu +: a = a + {val1}",
                f"valu multiply_add: temp = a * (1 << shift) + 0  # shift via multiply",
                f"valu ^: a = a ^ temp",
                f"valu +: a = a + {val3}",
            ]
            # Or even better:
            stage_analysis["fused_implementation"] = [
                f"valu +: a = a + {val1 + val3}  # combine constants",
                f"valu multiply_add: temp = a * {1 << 16} + 0  # if shift is 16",
                f"valu ^: a = a ^ temp",
            ]
            stage_analysis["savings"] = naive_ops - 3
        else:
            stage_analysis["fused_implementation"] = stage_analysis["naive_implementation"]
            stage_analysis["savings"] = 0
        
        total_ops_fused += len(stage_analysis["fused_implementation"])
        fusion_analysis.append(stage_analysis)
    
    # The REAL optimization: Use multiply_add for the entire pattern
    # Pattern: a = ((a + c1) ^ ((a + c1) << shift)) + c3
    # Let b = a + c1
    # Then: result = (b ^ (b << shift)) + c3
    # 
    # multiply_add can compute: dest = src1 * src2 + src3
    # So: b << shift = b * (2^shift) = multiply_add(b, 2^shift, 0)
    
    optimal_per_stage = {
        "instructions": [
            "valu +: b = a + c1           # 1 instruction",
            "valu multiply_add: temp = b * (2^shift) + 0  # 1 instruction (shift via multiply)",
            "valu ^: b = b ^ temp         # 1 instruction",
            "valu +: result = b + c3      # 1 instruction",
        ],
        "total_per_stage": 4,
        "or_with_constant_folding": [
            "valu +: b = a + (c1 + c3)    # fold constants",
            "valu multiply_add: temp = a * (2^shift) + 0",
            "valu ^: result = b ^ temp",
        ],
        "optimized_total": 3,
    }
    
    return {
        "hash_stages": len(HASH_STAGES),
        "stage_analysis": fusion_analysis,
        "summary": {
            "naive_ops_per_stage": total_ops_naive // len(HASH_STAGES),
            "fused_ops_per_stage": total_ops_fused // len(HASH_STAGES),
            "optimal_ops_per_stage": 3,  # With multiply_add for shift
            "naive_total": total_ops_naive,
            "optimal_total": 3 * len(HASH_STAGES),  # 18 ops for 6 stages
        },
        "key_insights": [
            "multiply_add(dest, a, 2^n, 0) computes a << n in one slot",
            "This frees the ALU shift slots for address calculation",
            "Constant folding: (a + c1) + c3 at end = a + (c1 + c3) if XOR doesn't interfere",
            "Best case: 3 VALU ops per hash stage (add, multiply_add for shift, xor)",
        ],
        "CRITICAL_CONSTRAINT": {
            "note": "multiply_add operands must be scratch addresses, NOT immediates!",
            "detail": "To use multiply_add(dest, a, multiplier, offset), you must first:",
            "steps": [
                "1. Use 'const' to load multiplier (e.g., 2^5 = 32) into scratch once during init",
                "2. Use 'const' to load offset constants into scratch once during init",
                "3. Then reference those scratch addresses in multiply_add",
            ],
            "setup_cost": "~6-12 cycles to load all shift multipliers (2^5, 2^13, etc.) during init",
            "example": "const(scratch[100], 32)  # 2^5 = 32, then multiply_add(dest, a, 100, offset_addr)",
        },
        "recommendations": [
            "Pre-load shift multipliers (2^5, 2^13, etc.) into scratch during init phase",
            "Pre-load offset constants into scratch during init phase",
            "Setup cost: ~6-12 init cycles, but saves 2+ VALU slots per hash stage",
            f"Optimal: {3 * len(HASH_STAGES)} VALU ops for all hash stages vs naive {total_ops_naive}",
        ],
    }


def audit_kernel_hash_implementation(
    kernel_module: str = "perf_takehome",
    tree_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
):
    """
    Audit a kernel's actual hash implementation for fusion opportunities.
    """
    if kernel_module == "perf_takehome":
        from perf_takehome import KernelBuilder
    elif kernel_module == "takehome_diff":
        from takehome_diff import KernelBuilder
    else:
        return {"error": f"Unknown kernel: {kernel_module}"}
    
    import random
    from problem import Tree
    
    random.seed(123)
    forest = Tree.generate(tree_height)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), batch_size, rounds)
    
    instrs = kb.instrs
    
    # Count multiply_add usage
    multiply_add_count = 0
    shift_count = 0
    add_count = 0
    xor_count = 0
    
    for instr in instrs:
        for valu_op in instr.get('valu', []):
            if valu_op[0] == 'multiply_add':
                multiply_add_count += 1
            elif valu_op[0] == '<<':
                shift_count += 1
            elif valu_op[0] == '+':
                add_count += 1
            elif valu_op[0] == '^':
                xor_count += 1
    
    # Expected if using multiply_add for shifts
    n_vectors = batch_size // 8
    hash_calls = rounds * n_vectors  # Approximate
    expected_multiply_add_if_optimal = len(HASH_STAGES) * hash_calls // 6  # Per group
    
    return {
        "multiply_add_count": multiply_add_count,
        "explicit_shift_count": shift_count,
        "add_count": add_count,
        "xor_count": xor_count,
        "using_multiply_add_for_hash": multiply_add_count > 100,
        "fusion_quality": "GOOD" if multiply_add_count > shift_count else "POOR",
        "recommendation": "Convert explicit shifts to multiply_add" if shift_count > multiply_add_count 
                         else "Hash implementation uses multiply_add efficiently",
    }


if __name__ == "__main__":
    print("=== Hash Fusion Opportunity Analysis ===")
    result = analyze_hash_fusion_opportunities()
    print(f"Hash stages: {result['hash_stages']}")
    print(f"\nSummary:")
    for k, v in result['summary'].items():
        print(f"  {k}: {v}")
    print(f"\nKey insights:")
    for insight in result['key_insights']:
        print(f"  - {insight}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    
    print("\n=== Kernel Hash Audit: perf_takehome ===")
    audit = audit_kernel_hash_implementation("perf_takehome")
    for k, v in audit.items():
        print(f"  {k}: {v}")
