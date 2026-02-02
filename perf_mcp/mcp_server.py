#!/usr/bin/env python3
"""
MCP Server for Performance Analysis Tools
==========================================
Exposes all performance analysis tools as MCP resources and tools.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add parent AND current to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

from trace_mcp import analyze_trace, parse_trace_json, run_and_trace
from kernel_diff_mcp import compare_kernels
from round_attribution_mcp import attribute_cycles_to_rounds
from slot_packing_mcp import analyze_slot_packing
from gather_pressure_mcp import analyze_gather_pressure
from tail_drain_mcp import analyze_tail_drain
from hash_stage_mcp import analyze_hash_stages
from address_lifetime_mcp import analyze_address_lifetimes

# New analysis tools
from scratch_budget_mcp import analyze_scratch_budget, format_scratch_budget
from critical_path_mcp import analyze_critical_path, format_critical_path
from mux_load_tradeoff_mcp import analyze_mux_load_tradeoff, format_mux_load_tradeoff
from instruction_entropy_mcp import analyze_instruction_entropy, format_instruction_entropy
from kernel_compare_mcp import compare_kernels as compare_kernel_impls, format_kernel_comparison

# Additional helper tools
from quick_test_mcp import quick_test, compare_kernels as quick_compare_kernels
from interleave_mcp import analyze_interleave, find_gather_phases
from round_breakdown_mcp import analyze_round_instructions, calculate_theoretical_minimum
from mux_tradeoff_mcp import analyze_mux_vs_load, estimate_cycle_savings

# Debug and exploration tools
from instr_diff_mcp import diff_instructions, find_divergence
from bottleneck_mcp import identify_bottleneck, format_bottleneck_report
from what_if_mcp import what_if, explore_optimizations, format_what_if_report
from register_liveness_mcp import analyze_register_liveness

# Advanced micro-architectural analysis tools
from cross_round_pipeline_mcp import analyze_round_transitions, calculate_round_overlap
from algebraic_fusion_mcp import analyze_hash_fusion_opportunities, audit_kernel_hash_implementation
from scratch_liveness_mcp import analyze_scratch_liveness, visualize_scratch_density
from dependency_latency_mcp import analyze_dependency_latency, check_address_load_timing
from optimal_strategy_mcp import calculate_optimal_strategy

# Initialize MCP server
server = Server("perf-analysis")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available performance analysis tools."""
    return [
        Tool(
            name="trace_analyze",
            description="Analyze trace.json file and compute utilization metrics. Returns cycle count, engine utilization (load/valu/alu/store), bubble cycles, tail drain cycles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_path": {
                        "type": "string",
                        "description": "Path to trace.json or trace.json.gz file"
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Print detailed per-cycle statistics",
                        "default": False
                    }
                },
                "required": ["trace_path"]
            }
        ),
        Tool(
            name="trace_run",
            description="Run kernel and generate trace.json file. Optionally analyze the generated trace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Output path for trace.json",
                        "default": "trace.json"
                    },
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "tree_height": {"type": "integer", "default": 10},
                    "analyze": {
                        "type": "boolean",
                        "description": "Analyze trace after running",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="kernel_diff",
            description="Compare two kernel trace versions and show cycle delta, utilization differences, and code changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_a": {
                        "type": "string",
                        "description": "Path to first trace file"
                    },
                    "trace_b": {
                        "type": "string",
                        "description": "Path to second trace file"
                    },
                    "label_a": {
                        "type": "string",
                        "default": "Version A"
                    },
                    "label_b": {
                        "type": "string",
                        "default": "Version B"
                    }
                },
                "required": ["trace_a", "trace_b"]
            }
        ),
        Tool(
            name="round_attribution",
            description="Attribute cycles to rounds and groups. Shows initialization cycles, per-round breakdown, and identifies slowest rounds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_path": {
                        "type": "string",
                        "description": "Path to trace.json file"
                    },
                    "num_rounds": {"type": "integer", "default": 16},
                    "vectors_per_round": {"type": "integer", "default": 32},
                    "num_parallel": {"type": "integer", "default": 6}
                },
                "required": ["trace_path"]
            }
        ),
        Tool(
            name="slot_packing",
            description="Analyze instruction slot packing efficiency. Shows per-engine utilization and overall efficiency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "tree_height": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="gather_pressure",
            description="Analyze gather/memory pressure. Counts scalar loads, repeated loads, and address recomputations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "tree_height": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="tail_drain",
            description="Analyze tail and drain phases. Shows cycles after last gather, store overlap efficiency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trace_path": {
                        "type": "string",
                        "description": "Path to trace.json file"
                    }
                },
                "required": ["trace_path"]
            }
        ),
        Tool(
            name="hash_stages",
            description="Analyze hash computation stages. Shows XOR operations, collapsed multiply_add sequences, hash/load overlap.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "tree_height": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="address_lifetimes",
            description="Analyze address register lifetimes. Tracks when addresses are computed and last used.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "tree_height": {"type": "integer", "default": 10}
                }
            }
        ),
        # === NEW TOOLS ===
        Tool(
            name="scratch_budget",
            description="Analyze scratch memory budget and capacity. Shows peak usage, remaining space, max preload depth, and recommendations for memory allocation strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "default": 256},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "current_preload_depth": {
                        "type": "integer",
                        "default": 2,
                        "description": "Current tree depth being preloaded (0=root, 1=level1, 2=level2)"
                    },
                    "num_parallel": {
                        "type": "integer",
                        "default": 6,
                        "description": "Number of vectors processed in parallel"
                    }
                }
            }
        ),
        Tool(
            name="critical_path",
            description="Analyze instruction dependencies and pipeline hazards. Identifies RAW hazards, dependency chains, and interleaving opportunities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {
                        "type": "string",
                        "default": "perf_takehome",
                        "description": "Python module containing the kernel (e.g., 'perf_takehome' or 'takehome_diff')"
                    },
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="mux_load_tradeoff",
            description="Calculate when to use arithmetic muxing vs memory gathering for tree node selection. Shows break-even point and per-level recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tree_height": {"type": "integer", "default": 10},
                    "batch_size": {"type": "integer", "default": 256},
                    "num_parallel": {"type": "integer", "default": 6},
                    "hash_cycles_per_group": {
                        "type": "integer",
                        "default": 18,
                        "description": "Typical hash computation cycles per group"
                    }
                }
            }
        ),
        Tool(
            name="instruction_entropy",
            description="Analyze slot competition and engine bottlenecks. Shows when engines are oversubscribed and identifies the true throughput bottleneck.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {
                        "type": "string",
                        "default": "perf_takehome",
                        "description": "Python module containing the kernel"
                    },
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="kernel_compare",
            description="Compare two kernel implementations. Shows cycle count differences, interleaving effectiveness, and identifies what changed between versions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_a": {
                        "type": "string",
                        "default": "perf_takehome",
                        "description": "First kernel module name"
                    },
                    "kernel_b": {
                        "type": "string",
                        "default": "takehome_diff",
                        "description": "Second kernel module name"
                    },
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        # === QUICK TESTING TOOLS ===
        Tool(
            name="quick_test",
            description="Run a kernel and get pass/fail + key metrics in one call. Perfect for fast iteration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {
                        "type": "string",
                        "default": "perf_takehome",
                        "description": "Kernel module: 'perf_takehome' or 'takehome_diff'"
                    },
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "seed": {"type": "integer", "default": 123}
                }
            }
        ),
        Tool(
            name="quick_compare",
            description="Compare perf_takehome vs takehome_diff side-by-side. Shows pass/fail, cycles, and which is better.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "seed": {"type": "integer", "default": 123}
                }
            }
        ),
        # === INTERLEAVING ANALYSIS ===
        Tool(
            name="analyze_interleave",
            description="Analyze how well load and valu operations are interleaved. Shows pattern breakdown and missed opportunities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {
                        "type": "string",
                        "default": "perf_takehome",
                        "description": "Kernel module name"
                    },
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256},
                    "show_instructions": {
                        "type": "integer",
                        "default": 30,
                        "description": "Number of instructions to show in sample"
                    },
                    "start_from": {
                        "type": "integer",
                        "default": 0,
                        "description": "Starting instruction index for sample"
                    }
                }
            }
        ),
        Tool(
            name="find_gather_phases",
            description="Identify scatter-gather phases and analyze their efficiency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {
                        "type": "string",
                        "default": "perf_takehome"
                    },
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        # === ROUND ANALYSIS ===
        Tool(
            name="round_breakdown",
            description="Analyze instruction count per round. Shows init/main/store phases and estimated cycles per round type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {
                        "type": "string",
                        "default": "perf_takehome"
                    },
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="theoretical_minimum",
            description="Calculate the theoretical minimum cycles for the kernel. Shows where the floor is.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        # === MUX VS LOAD STRATEGY ===
        Tool(
            name="mux_vs_load",
            description="Calculate when to use arithmetic muxing vs scatter-gather loads. Shows cost per tree level.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tree_height": {"type": "integer", "default": 10},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        # === DEBUG & EXPLORATION TOOLS ===
        Tool(
            name="instr_diff",
            description="Compare instruction streams from two kernels side by side. Shows where they diverge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_a": {"type": "string", "default": "perf_takehome"},
                    "kernel_b": {"type": "string", "default": "takehome_diff"},
                    "start": {"type": "integer", "default": 0, "description": "Start instruction index"},
                    "count": {"type": "integer", "default": 50, "description": "Number of instructions to show"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="find_divergence",
            description="Find where two kernel instruction streams first diverge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_a": {"type": "string", "default": "perf_takehome"},
                    "kernel_b": {"type": "string", "default": "takehome_diff"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="bottleneck",
            description="Identify THE primary bottleneck in a kernel. Returns diagnosis with evidence and fix recommendation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="what_if",
            description="Calculate estimated cycles under hypothetical changes (more loads/cycle, more mux rounds, etc).",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_cycles": {"type": "integer", "default": 2512},
                    "loads_per_cycle": {"type": "integer", "description": "Hypothetical loads per cycle (default 2)"},
                    "valus_per_cycle": {"type": "integer", "description": "Hypothetical valus per cycle (default 6)"},
                    "mux_rounds": {"type": "integer", "description": "Number of rounds to mux (default 3)"},
                    "perfect_interleave": {"type": "boolean", "default": False}
                }
            }
        ),
        Tool(
            name="explore_optimizations",
            description="Explore various optimization scenarios and their potential impact.",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_cycles": {"type": "integer", "default": 2512}
                }
            }
        ),
        Tool(
            name="register_liveness",
            description="Analyze vector register usage and liveness. Shows peak register pressure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        # Advanced micro-architectural analysis tools
        Tool(
            name="cross_round_pipeline",
            description="Analyze round-transition bubbles and pipeline efficiency. Shows where Load is idle waiting for VALU (or vice versa) at round boundaries. Key for achieving 'single continuous pipe' scheduling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="round_overlap_score",
            description="Calculate overlap score: percentage of instructions that combine Load+VALU operations. Higher = better pipelining. Target >60% for sub-1400 cycles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="analyze_fusion_opportunities",
            description="Analyze HASH_STAGES for multiply_add fusion opportunities. Shows how to collapse multiple hash ops into single instructions. Key optimization for freeing VALU slots.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="audit_hash_implementation",
            description="Audit a kernel's actual hash implementation. Counts multiply_add usage vs explicit shifts to measure fusion quality.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="scratch_liveness",
            description="Map scratch address usage over time. Shows 'dead zones' where addresses are allocated but unused. Key for triple buffering and scratch aliasing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="visualize_scratch_density",
            description="Visualize scratch memory pressure timeline. Shows ASCII heatmap of memory usage over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="dependency_latency",
            description="Trace producer-consumer latency. Shows back-to-back (optimal) vs long-latency dependencies. Key for understanding data hazards.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="address_load_timing",
            description="Check timing between address calculation and memory loads. Optimal is gap=1 cycle. Large gaps waste scratch.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_module": {"type": "string", "default": "perf_takehome"},
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        ),
        Tool(
            name="optimal_strategy",
            description="Calculate optimal mux strategy considering tree wrapping. KEY INSIGHT: With tree_height=10 and rounds=16, tree wraps at round 10. Mux levels 0-6 reduces gather to only 3 rounds!",
            inputSchema={
                "type": "object",
                "properties": {
                    "tree_height": {"type": "integer", "default": 10},
                    "rounds": {"type": "integer", "default": 16},
                    "batch_size": {"type": "integer", "default": 256}
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "trace_analyze":
        trace_path = arguments["trace_path"]
        verbose = arguments.get("verbose", False)
        
        try:
            stats = analyze_trace(trace_path, verbose=verbose)
            result = {
                "total_cycles": stats.total_cycles,
                "load_utilization": round(stats.load_utilization, 2),
                "valu_utilization": round(stats.valu_utilization, 2),
                "alu_utilization": round(stats.alu_utilization, 2),
                "store_utilization": round(stats.store_utilization, 2),
                "bubble_cycles": stats.bubble_cycles,
                "tail_drain_cycles": stats.tail_drain_cycles,
                "last_gather_cycle": stats.last_gather_cycle,
                "first_store_cycle": stats.first_store_cycle
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "trace_run":
        output_path = arguments.get("output_path", "trace.json")
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        tree_height = arguments.get("tree_height", 10)
        analyze = arguments.get("analyze", False)
        
        try:
            cycles = run_and_trace(output_path, rounds, batch_size, tree_height)
            result = {"cycles": cycles, "output_path": output_path}
            
            if analyze:
                stats = analyze_trace(output_path, verbose=False)
                result.update({
                    "load_utilization": round(stats.load_utilization, 2),
                    "valu_utilization": round(stats.valu_utilization, 2),
                    "tail_drain_cycles": stats.tail_drain_cycles
                })
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "kernel_diff":
        trace_a = arguments["trace_a"]
        trace_b = arguments["trace_b"]
        label_a = arguments.get("label_a", "Version A")
        label_b = arguments.get("label_b", "Version B")
        
        try:
            # Capture output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                compare_kernels(trace_a, trace_b, label_a, label_b)
            
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "round_attribution":
        trace_path = arguments["trace_path"]
        num_rounds = arguments.get("num_rounds", 16)
        vectors_per_round = arguments.get("vectors_per_round", 32)
        num_parallel = arguments.get("num_parallel", 6)
        
        try:
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                attribute_cycles_to_rounds(
                    trace_path,
                    rounds=num_rounds,
                    n_vectors=vectors_per_round,
                    num_parallel=num_parallel,
                    verbose=True
                )
            
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "slot_packing":
        from perf_takehome import KernelBuilder
        from problem import Tree, Input, build_mem_image
        
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        tree_height = arguments.get("tree_height", 10)
        
        try:
            import random
            random.seed(123)
            forest = Tree.generate(tree_height)
            inp = Input.generate(forest, batch_size, rounds)
            
            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                analyze_slot_packing(kb.instrs, verbose=True)
            
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "gather_pressure":
        from perf_takehome import KernelBuilder
        from problem import Tree, Input, build_mem_image
        
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        tree_height = arguments.get("tree_height", 10)
        
        try:
            import random
            random.seed(123)
            forest = Tree.generate(tree_height)
            inp = Input.generate(forest, batch_size, rounds)
            
            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                analyze_gather_pressure(kb.instrs, verbose=True)
            
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "tail_drain":
        trace_path = arguments["trace_path"]
        
        try:
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                analyze_tail_drain(trace_path, verbose=True)
            
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "hash_stages":
        from perf_takehome import KernelBuilder
        from problem import Tree, Input, build_mem_image
        
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        tree_height = arguments.get("tree_height", 10)
        
        try:
            import random
            random.seed(123)
            forest = Tree.generate(tree_height)
            inp = Input.generate(forest, batch_size, rounds)
            
            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                analyze_hash_stages(kb.instrs, verbose=True)
            
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "address_lifetimes":
        from perf_takehome import KernelBuilder
        from problem import Tree, Input, build_mem_image
        
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        tree_height = arguments.get("tree_height", 10)
        
        try:
            import random
            random.seed(123)
            forest = Tree.generate(tree_height)
            inp = Input.generate(forest, batch_size, rounds)
            
            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                analyze_address_lifetimes(kb.instrs, verbose=True)
            
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    # === NEW TOOL HANDLERS ===
    
    elif name == "scratch_budget":
        batch_size = arguments.get("batch_size", 256)
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        current_preload_depth = arguments.get("current_preload_depth", 2)
        num_parallel = arguments.get("num_parallel", 6)
        
        try:
            budget = analyze_scratch_budget(
                batch_size=batch_size,
                tree_height=tree_height,
                rounds=rounds,
                current_preload_depth=current_preload_depth,
                num_parallel=num_parallel
            )
            return [TextContent(type="text", text=format_scratch_budget(budget))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "critical_path":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            import importlib
            mod = importlib.import_module(kernel_module)
            
            kb = mod.KernelBuilder()
            # CORRECT: Tree has 2^(height+1) - 1 nodes
            n_nodes = 2 ** (tree_height + 1) - 1
            kb.build_kernel(tree_height, n_nodes, batch_size, rounds)
            
            analysis = analyze_critical_path(kb.instrs)
            return [TextContent(type="text", text=format_critical_path(analysis))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "mux_load_tradeoff":
        tree_height = arguments.get("tree_height", 10)
        batch_size = arguments.get("batch_size", 256)
        num_parallel = arguments.get("num_parallel", 6)
        hash_cycles = arguments.get("hash_cycles_per_group", 18)
        
        try:
            analysis = analyze_mux_load_tradeoff(
                tree_height=tree_height,
                batch_size=batch_size,
                num_parallel=num_parallel,
                hash_cycles_per_group=hash_cycles
            )
            return [TextContent(type="text", text=format_mux_load_tradeoff(analysis))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "instruction_entropy":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            import importlib
            mod = importlib.import_module(kernel_module)
            
            kb = mod.KernelBuilder()
            # CORRECT: Tree has 2^(height+1) - 1 nodes
            n_nodes = 2 ** (tree_height + 1) - 1
            kb.build_kernel(tree_height, n_nodes, batch_size, rounds)
            
            analysis = analyze_instruction_entropy(kb.instrs)
            return [TextContent(type="text", text=format_instruction_entropy(analysis))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "kernel_compare":
        kernel_a = arguments.get("kernel_a", "perf_takehome")
        kernel_b = arguments.get("kernel_b", "takehome_diff")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            comparison = compare_kernel_impls(
                kernel_a, kernel_b,
                tree_height=tree_height,
                rounds=rounds,
                batch_size=batch_size
            )
            return [TextContent(type="text", text=format_kernel_comparison(comparison))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    # === NEW QUICK TESTING TOOLS ===
    
    elif name == "quick_test":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        seed = arguments.get("seed", 123)
        
        try:
            result = quick_test(kernel_module, tree_height, rounds, batch_size, seed)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "quick_compare":
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        seed = arguments.get("seed", 123)
        
        try:
            result = quick_compare_kernels(tree_height, rounds, batch_size, seed)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "analyze_interleave":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        show_instructions = arguments.get("show_instructions", 30)
        start_from = arguments.get("start_from", 0)
        
        try:
            result = analyze_interleave(kernel_module, tree_height, rounds, batch_size, show_instructions, start_from)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "find_gather_phases":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = find_gather_phases(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "round_breakdown":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = analyze_round_instructions(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "theoretical_minimum":
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = calculate_theoretical_minimum(tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "mux_vs_load":
        tree_height = arguments.get("tree_height", 10)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = analyze_mux_vs_load(tree_height, batch_size)
            # Format nicely
            lines = [
                "=== MUX VS LOAD ANALYSIS ===",
                "",
                "Level | Nodes | Mux Ops | Mux Cycles | Load Cycles | Winner | Savings | Scratch",
                "-" * 85
            ]
            for r in result["analysis"][:8]:
                lines.append(f"  {r['level']:3d} | {r['n_nodes']:5d} | {r['mux_valu_ops']:7d} | {r['mux_cycles']:10d} | {r['load_cycles']:11d} | {r['winner']:6s} | {r['savings']:7d} | {r['mux_scratch']:7d}")
            lines.append("")
            lines.append(f"Recommended MUX levels: {result['recommended_mux_levels']}")
            lines.append(f"Recommended LOAD levels: {result['recommended_load_levels']}")
            lines.append(f"Total scratch for MUX: {result['mux_scratch_needed']}")
            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    # === DEBUG & EXPLORATION TOOLS ===
    
    elif name == "instr_diff":
        kernel_a = arguments.get("kernel_a", "perf_takehome")
        kernel_b = arguments.get("kernel_b", "takehome_diff")
        start = arguments.get("start", 0)
        count = arguments.get("count", 50)
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = diff_instructions(kernel_a, kernel_b, start, count, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=result["comparison"])]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "find_divergence":
        kernel_a = arguments.get("kernel_a", "perf_takehome")
        kernel_b = arguments.get("kernel_b", "takehome_diff")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = find_divergence(kernel_a, kernel_b, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "bottleneck":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = identify_bottleneck(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=format_bottleneck_report(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "what_if":
        current_cycles = arguments.get("current_cycles", 2512)
        loads_per_cycle = arguments.get("loads_per_cycle")
        valus_per_cycle = arguments.get("valus_per_cycle")
        mux_rounds = arguments.get("mux_rounds")
        perfect_interleave = arguments.get("perfect_interleave", False)
        
        try:
            result = what_if(
                current_cycles=current_cycles,
                loads_per_cycle=loads_per_cycle,
                valus_per_cycle=valus_per_cycle,
                mux_rounds=mux_rounds,
                perfect_interleave=perfect_interleave
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "explore_optimizations":
        current_cycles = arguments.get("current_cycles", 2512)
        
        try:
            scenarios = explore_optimizations(current_cycles)
            return [TextContent(type="text", text=format_what_if_report(scenarios))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "register_liveness":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = analyze_register_liveness(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    # Advanced micro-architectural tools
    elif name == "cross_round_pipeline":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = analyze_round_transitions(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "round_overlap_score":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = calculate_round_overlap(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "analyze_fusion_opportunities":
        try:
            result = analyze_hash_fusion_opportunities()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "audit_hash_implementation":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = audit_kernel_hash_implementation(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "scratch_liveness":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = analyze_scratch_liveness(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "visualize_scratch_density":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                visualize_scratch_density(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=f.getvalue())]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "dependency_latency":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = analyze_dependency_latency(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "address_load_timing":
        kernel_module = arguments.get("kernel_module", "perf_takehome")
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = check_address_load_timing(kernel_module, tree_height, rounds, batch_size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "optimal_strategy":
        tree_height = arguments.get("tree_height", 10)
        rounds = arguments.get("rounds", 16)
        batch_size = arguments.get("batch_size", 256)
        
        try:
            result = calculate_optimal_strategy(tree_height, rounds, batch_size)
            # Format the key findings
            opt = result['optimal_strategy']
            summary = {
                "round_to_level": result['round_to_level'],
                "tree_wraps_at_round": result['tree_wraps_at_round'],
                "optimal_mux_depth": opt['mux_depth'],
                "optimal_mux_nodes": opt['mux_nodes'],
                "optimal_scratch": opt['mux_scratch'],
                "gather_rounds_needed": opt['gather_rounds'],
                "which_rounds_need_gather": opt['which_gather_rounds'],
                "estimated_cycles": opt['total_cycles'],
                "key_insight": result['improvement_potential']['key_insight'],
                "all_strategies": result['all_strategies'],
            }
            return [TextContent(type="text", text=json.dumps(summary, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
