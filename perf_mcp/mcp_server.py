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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
