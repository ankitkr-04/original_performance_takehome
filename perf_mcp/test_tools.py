#!/usr/bin/env python3
"""
Test suite to verify all MCP tools are working correctly.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all MCP modules can be imported"""
    print("Testing imports...")
    
    try:
        import trace_mcp
        print("  ✓ trace_mcp")
    except Exception as e:
        print(f"  ✗ trace_mcp: {e}")
        return False
    
    try:
        import kernel_diff_mcp
        print("  ✓ kernel_diff_mcp")
    except Exception as e:
        print(f"  ✗ kernel_diff_mcp: {e}")
        return False
    
    try:
        import round_attribution_mcp
        print("  ✓ round_attribution_mcp")
    except Exception as e:
        print(f"  ✗ round_attribution_mcp: {e}")
        return False
    
    try:
        import slot_packing_mcp
        print("  ✓ slot_packing_mcp")
    except Exception as e:
        print(f"  ✗ slot_packing_mcp: {e}")
        return False
    
    try:
        import gather_pressure_mcp
        print("  ✓ gather_pressure_mcp")
    except Exception as e:
        print(f"  ✗ gather_pressure_mcp: {e}")
        return False
    
    try:
        import tail_drain_mcp
        print("  ✓ tail_drain_mcp")
    except Exception as e:
        print(f"  ✗ tail_drain_mcp: {e}")
        return False
    
    try:
        import param_sweep_mcp
        print("  ✓ param_sweep_mcp")
    except Exception as e:
        print(f"  ✗ param_sweep_mcp: {e}")
        return False
    
    try:
        import hash_stage_mcp
        print("  ✓ hash_stage_mcp")
    except Exception as e:
        print(f"  ✗ hash_stage_mcp: {e}")
        return False
    
    try:
        import address_lifetime_mcp
        print("  ✓ address_lifetime_mcp")
    except Exception as e:
        print(f"  ✗ address_lifetime_mcp: {e}")
        return False
    
    return True


def test_static_analysis():
    """Test static analysis tools that don't need trace.json"""
    print("\nTesting static analysis tools...")
    
    from perf_takehome import KernelBuilder
    from problem import Tree
    import random
    
    random.seed(123)
    forest = Tree.generate(10)
    
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), 256, 16)
    
    try:
        from slot_packing_mcp import analyze_slot_packing
        result = analyze_slot_packing(kb.instrs, verbose=False)
        assert 'avg_load' in result
        print("  ✓ Slot packing analysis")
    except Exception as e:
        print(f"  ✗ Slot packing analysis: {e}")
        return False
    
    try:
        from gather_pressure_mcp import analyze_gather_pressure
        result = analyze_gather_pressure(kb.instrs, verbose=False)
        assert 'scalar_loads' in result
        print("  ✓ Gather pressure analysis")
    except Exception as e:
        print(f"  ✗ Gather pressure analysis: {e}")
        return False
    
    try:
        from hash_stage_mcp import analyze_hash_stages
        result = analyze_hash_stages(kb.instrs, verbose=False)
        assert 'collapsed_stages' in result
        print("  ✓ Hash stage analysis")
    except Exception as e:
        print(f"  ✗ Hash stage analysis: {e}")
        return False
    
    try:
        from address_lifetime_mcp import analyze_address_lifetimes
        result = analyze_address_lifetimes(kb.instrs, verbose=False)
        assert 'total_addresses' in result
        print("  ✓ Address lifetime analysis")
    except Exception as e:
        print(f"  ✗ Address lifetime analysis: {e}")
        return False
    
    return True


def test_trace_generation():
    """Test trace generation"""
    print("\nTesting trace generation...")
    
    try:
        from trace_mcp import run_and_trace
        cycles = run_and_trace(
            forest_height=10,
            rounds=16,
            batch_size=256,
            output_path="test_trace.json",
            seed=123
        )
        assert cycles > 0
        print(f"  ✓ Trace generation (cycles: {cycles})")
        
        # Clean up
        import os
        if os.path.exists("test_trace.json"):
            os.remove("test_trace.json")
        
        return True
    except Exception as e:
        print(f"  ✗ Trace generation: {e}")
        return False


def main():
    print("="*70)
    print("MCP TOOLS TEST SUITE")
    print("="*70)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
        print("\n⚠ Some imports failed")
    else:
        print("\n✓ All imports successful")
    
    # Test static analysis
    if not test_static_analysis():
        success = False
        print("\n⚠ Some static analysis tools failed")
    else:
        print("\n✓ All static analysis tools working")
    
    # Test trace generation (optional - takes time)
    print("\n" + "="*70)
    response = input("Run trace generation test? (takes ~10 seconds) [y/N]: ")
    if response.lower() == 'y':
        if not test_trace_generation():
            success = False
            print("\n⚠ Trace generation failed")
        else:
            print("\n✓ Trace generation working")
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED")
        print("\nYour MCP tooling suite is ready to use!")
        print("\nQuick start:")
        print("  python mcp/server.py --help")
        print("  python mcp/server.py trace run --analyze")
        print("  python mcp/server.py slot-packing")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nCheck the error messages above for details.")
    print("="*70)


if __name__ == "__main__":
    main()
