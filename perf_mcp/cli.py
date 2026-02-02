#!/usr/bin/env python3
"""
Performance MCP Server - Main Entry Point
==========================================
Unified interface for all performance analysis tools.
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Performance MCP Tooling Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:

TIER 1 - Must Have Tools (Unlock Last 1000 Cycles):
  trace          Run and analyze kernel traces
  diff           Compare two kernel versions
  attribution    Attribute cycles to rounds and groups

TIER 2 - High ROI Tools (Make Tier-1 Stronger):
  slot-packing   Analyze instruction slot packing efficiency
  gather         Analyze gather/memory pressure
  tail           Analyze tail and drain phases

TIER 3 - Exploration & Auto-tuning:
  sweep          Sweep parameters to find optimal config
  hash           Analyze hash stage costs and optimizations
  address        Analyze address register lifetimes

Examples:
  # Run kernel with trace and analyze
  python mcp/server.py trace run --analyze
  
  # Compare two traces
  python mcp/server.py diff trace_v1.json trace_v2.json
  
  # Analyze slot packing
  python mcp/server.py slot-packing
  
  # Parameter sweep
  python mcp/server.py sweep --num-parallel 4,5,6,7
  
  # Full analysis pipeline
  python mcp/server.py trace run --analyze --csv results.csv
  python mcp/server.py attribution trace.json
  python mcp/server.py tail trace.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='tool', help='Analysis tool to run')
    
    # Trace MCP
    trace_parser = subparsers.add_parser('trace', help='Trace analysis (Tier 1)')
    trace_parser.add_argument('action', choices=['run', 'analyze'], help='Run kernel or analyze existing trace')
    trace_parser.add_argument('trace_path', nargs='?', help='Path to trace.json (for analyze)')
    trace_parser.add_argument('--rounds', type=int, default=16)
    trace_parser.add_argument('--batch', type=int, default=256)
    trace_parser.add_argument('--height', type=int, default=10)
    trace_parser.add_argument('--output', '-o', default='trace.json')
    trace_parser.add_argument('--analyze', action='store_true', help='Analyze after running')
    trace_parser.add_argument('--csv', help='Export to CSV')
    trace_parser.add_argument('--verbose', '-v', action='store_true')
    
    # Diff MCP
    diff_parser = subparsers.add_parser('diff', help='Kernel comparison (Tier 1)')
    diff_parser.add_argument('trace_a', help='First trace file')
    diff_parser.add_argument('trace_b', help='Second trace file')
    diff_parser.add_argument('--label-a', default='Version A')
    diff_parser.add_argument('--label-b', default='Version B')
    diff_parser.add_argument('--json', help='Export to JSON')
    
    # Attribution MCP
    attr_parser = subparsers.add_parser('attribution', help='Round attribution (Tier 1)')
    attr_parser.add_argument('trace_path', help='Path to trace.json')
    attr_parser.add_argument('--rounds', type=int, default=16)
    attr_parser.add_argument('--vectors', type=int, default=32)
    attr_parser.add_argument('--parallel', type=int, default=6)
    attr_parser.add_argument('--csv', help='Export to CSV')
    
    # Slot Packing MCP
    slot_parser = subparsers.add_parser('slot-packing', help='Slot packing analysis (Tier 2)')
    slot_parser.add_argument('--rounds', type=int, default=16)
    slot_parser.add_argument('--batch', type=int, default=256)
    slot_parser.add_argument('--height', type=int, default=10)
    
    # Gather Pressure MCP
    gather_parser = subparsers.add_parser('gather', help='Gather pressure analysis (Tier 2)')
    gather_parser.add_argument('--rounds', type=int, default=16)
    gather_parser.add_argument('--batch', type=int, default=256)
    gather_parser.add_argument('--height', type=int, default=10)
    
    # Tail/Drain MCP
    tail_parser = subparsers.add_parser('tail', help='Tail/drain analysis (Tier 2)')
    tail_parser.add_argument('trace_path', help='Path to trace.json')
    
    # Parameter Sweep MCP
    sweep_parser = subparsers.add_parser('sweep', help='Parameter sweep (Tier 3)')
    sweep_parser.add_argument('--rounds', type=int, default=16)
    sweep_parser.add_argument('--batch', type=int, default=256)
    sweep_parser.add_argument('--height', type=int, default=10)
    sweep_parser.add_argument('--num-parallel', help='e.g., 4,5,6')
    sweep_parser.add_argument('--json', help='Export results to JSON')
    
    # Hash Stage MCP
    hash_parser = subparsers.add_parser('hash', help='Hash stage analysis (Tier 3)')
    hash_parser.add_argument('--rounds', type=int, default=16)
    hash_parser.add_argument('--batch', type=int, default=256)
    hash_parser.add_argument('--height', type=int, default=10)
    
    # Address Lifetime MCP
    addr_parser = subparsers.add_parser('address', help='Address lifetime analysis (Tier 3)')
    addr_parser.add_argument('--rounds', type=int, default=16)
    addr_parser.add_argument('--batch', type=int, default=256)
    addr_parser.add_argument('--height', type=int, default=10)
    
    args = parser.parse_args()
    
    if not args.tool:
        parser.print_help()
        return
    
    # Route to appropriate MCP
    if args.tool == 'trace':
        from trace_mcp import run_and_trace, analyze_trace, export_csv
        
        if args.action == 'run':
            cycles = run_and_trace(
                forest_height=args.height,
                rounds=args.rounds,
                batch_size=args.batch,
                output_path=args.output
            )
            
            if args.analyze:
                stats = analyze_trace(args.output, verbose=True)
                if args.csv:
                    export_csv(stats, args.csv)
        
        elif args.action == 'analyze':
            if not args.trace_path:
                print("Error: trace_path required for analyze action")
                return
            
            stats = analyze_trace(args.trace_path, verbose=args.verbose or True)
            if args.csv:
                export_csv(stats, args.csv)
    
    elif args.tool == 'diff':
        from kernel_diff_mcp import compare_kernels
        compare_kernels(args.trace_a, args.trace_b, args.label_a, args.label_b)
        
        if args.json:
            print(f"JSON export not yet implemented for diff")
    
    elif args.tool == 'attribution':
        from round_attribution_mcp import attribute_cycles_to_rounds
        attribute_cycles_to_rounds(
            args.trace_path,
            rounds=args.rounds,
            n_vectors=args.vectors,
            num_parallel=args.parallel,
            verbose=True
        )
    
    elif args.tool == 'slot-packing':
        from slot_packing_mcp import analyze_slot_packing
        from perf_takehome import KernelBuilder
        from problem import Tree
        import random
        
        random.seed(123)
        forest = Tree.generate(args.height)
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), args.batch, args.rounds)
        
        analyze_slot_packing(kb.instrs, verbose=True)
    
    elif args.tool == 'gather':
        from gather_pressure_mcp import analyze_gather_pressure
        from perf_takehome import KernelBuilder
        from problem import Tree
        import random
        
        random.seed(123)
        forest = Tree.generate(args.height)
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), args.batch, args.rounds)
        
        analyze_gather_pressure(kb.instrs, verbose=True)
    
    elif args.tool == 'tail':
        from tail_drain_mcp import analyze_tail_drain
        analyze_tail_drain(args.trace_path, verbose=True)
    
    elif args.tool == 'sweep':
        from param_sweep_mcp import sweep_parameters
        
        param_grid = {}
        if args.num_parallel:
            param_grid['NUM_PARALLEL'] = [int(x) for x in args.num_parallel.split(',')]
        else:
            param_grid['NUM_PARALLEL'] = [4, 5, 6, 7, 8]
        
        sweep_parameters(
            param_grid=param_grid,
            forest_height=args.height,
            rounds=args.rounds,
            batch_size=args.batch,
            verbose=True
        )
    
    elif args.tool == 'hash':
        from hash_stage_mcp import analyze_hash_stages
        from perf_takehome import KernelBuilder
        from problem import Tree
        import random
        
        random.seed(123)
        forest = Tree.generate(args.height)
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), args.batch, args.rounds)
        
        analyze_hash_stages(kb.instrs, verbose=True)
    
    elif args.tool == 'address':
        from address_lifetime_mcp import analyze_address_lifetimes
        from perf_takehome import KernelBuilder
        from problem import Tree
        import random
        
        random.seed(123)
        forest = Tree.generate(args.height)
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), args.batch, args.rounds)
        
        analyze_address_lifetimes(kb.instrs, verbose=True)


if __name__ == "__main__":
    main()
