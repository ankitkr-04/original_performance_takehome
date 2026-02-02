#!/usr/bin/env python3
"""
Tier 3 MCP: Parameter Sweep
=============================
Auto-rerun kernel with different parameters to find optimal configuration.
"""

import sys
from pathlib import Path
import itertools
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SweepResult:
    """Result for one parameter configuration"""
    params: Dict
    cycles: int
    success: bool
    error: str = ""


def sweep_parameters(
    param_grid: Dict[str, List],
    forest_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    seed: int = 123,
    verbose: bool = True
) -> List[SweepResult]:
    """
    Sweep through parameter combinations and measure performance.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        forest_height: Tree height
        rounds: Number of rounds
        batch_size: Batch size
        seed: Random seed
        verbose: Print progress
    
    Returns:
        List of SweepResult objects
    """
    import random
    from problem import Tree, Input, build_mem_image, Machine, N_CORES, reference_kernel2
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PARAMETER SWEEP")
        print(f"{'='*70}")
        print(f"\nTesting {len(combinations)} parameter combinations...")
        print(f"Parameters: {param_names}")
        print(f"\n{'='*70}")
    
    # Prepare test data once
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    base_mem = build_mem_image(forest, inp)
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        if verbose:
            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        try:
            # Import kernel builder - need to modify it to accept parameters
            # For now, we'll note that this requires kernel parameterization
            from perf_takehome import KernelBuilder
            
            kb = KernelBuilder()
            
            # Apply parameters (this is a template - actual kernel needs to support this)
            # For demonstration, we'll show the pattern
            mem = base_mem.copy()
            
            # Build kernel with these parameters
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
            
            # Run simulation
            machine = Machine(
                mem,
                kb.instrs,
                kb.debug_info(),
                n_cores=N_CORES,
                trace=False
            )
            machine.enable_pause = False
            machine.enable_debug = False
            
            for ref_mem in reference_kernel2(mem, {}):
                machine.run()
                inp_values_p = ref_mem[6]
                assert (
                    machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                    == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
                ), "Incorrect result"
            
            cycles = machine.cycle
            
            results.append(SweepResult(
                params=params,
                cycles=cycles,
                success=True
            ))
            
            if verbose:
                print(f"  ✓ Success: {cycles} cycles")
        
        except Exception as e:
            results.append(SweepResult(
                params=params,
                cycles=999999,
                success=False,
                error=str(e)
            ))
            
            if verbose:
                print(f"  ✗ Failed: {str(e)[:50]}")
    
    # Sort by cycles
    results.sort(key=lambda x: x.cycles)
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"\nTop 10 configurations:")
        print(f"\n{'Rank':<6} │ {'Cycles':<10} │ Parameters")
        print(f"{'-'*70}")
        
        for i, result in enumerate(results[:10]):
            if result.success:
                param_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
                print(f"{i+1:<6} │ {result.cycles:<10} │ {param_str}")
        
        # Best vs worst
        best = results[0]
        successful = [r for r in results if r.success]
        if len(successful) > 1:
            worst = successful[-1]
            improvement = ((worst.cycles - best.cycles) / worst.cycles * 100)
            
            print(f"\n{'='*70}")
            print(f"Best configuration: {best.params}")
            print(f"  Cycles: {best.cycles}")
            print(f"\nWorst configuration: {worst.params}")
            print(f"  Cycles: {worst.cycles}")
            print(f"\nImprovement: {improvement:.1f}% faster")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter Sweep MCP - Find optimal parameters")
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--height', type=int, default=10)
    parser.add_argument('--json', help='Export results to JSON')
    
    # Example parameter grid
    parser.add_argument('--num-parallel', help='Comma-separated list of NUM_PARALLEL values (e.g., 4,5,6)')
    parser.add_argument('--custom-grid', help='JSON file with custom parameter grid')
    
    args = parser.parse_args()
    
    # Build parameter grid
    param_grid = {}
    
    if args.num_parallel:
        param_grid['NUM_PARALLEL'] = [int(x) for x in args.num_parallel.split(',')]
    
    if args.custom_grid:
        with open(args.custom_grid, 'r') as f:
            param_grid = json.load(f)
    
    if not param_grid:
        # Default sweep
        print("No parameters specified. Using default sweep:")
        print("  NUM_PARALLEL: [4, 5, 6, 7, 8]")
        param_grid = {
            'NUM_PARALLEL': [4, 5, 6, 7, 8]
        }
    
    results = sweep_parameters(
        param_grid=param_grid,
        forest_height=args.height,
        rounds=args.rounds,
        batch_size=args.batch,
        verbose=True
    )
    
    if args.json:
        output = {
            'param_grid': param_grid,
            'results': [
                {
                    'params': r.params,
                    'cycles': r.cycles,
                    'success': r.success,
                    'error': r.error
                }
                for r in results
            ]
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results exported to: {args.json}")


if __name__ == "__main__":
    # Note: This MCP requires kernel modifications to be parameterizable
    print("\n⚠ NOTE: Parameter sweep requires modifying KernelBuilder to accept parameters")
    print("This is a template showing the pattern. Actual implementation needs:")
    print("  1. Parameterizable kernel builder")
    print("  2. Parameter-specific code generation")
    print("  3. Safe parameter validation\n")
    
    main()
