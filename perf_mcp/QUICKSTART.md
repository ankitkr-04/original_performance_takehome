# MCP Quick Start Guide

## Installation

No installation needed - just run from the mcp folder!

## Test the Tools

### 1. Run your first trace

```bash
cd /home/ankit/Documents/mastery/projects/original_performance_takehome
python mcp/server.py trace run --analyze
```

This will:

- Run your current kernel implementation
- Generate `trace.json`
- Automatically analyze it
- Show cycle count and utilization metrics

### 2. Try the analysis tools

```bash
# Analyze slot packing (instruction efficiency)
python mcp/server.py slot-packing

# Analyze memory access patterns
python mcp/server.py gather

# Analyze hash computation
python mcp/server.py hash

# Analyze address register usage
python mcp/server.py address
```

### 3. After making a change

```bash
# Save current trace
mv trace.json trace_before.json

# Run with new code
python mcp/server.py trace run --output trace_after.json

# Compare
python mcp/server.py diff trace_before.json trace_after.json
```

## Quick Command Reference

```bash
# Trace Analysis
python mcp/server.py trace run --analyze              # Run + analyze
python mcp/server.py trace analyze trace.json -v      # Analyze existing
python mcp/server.py trace run --csv stats.csv        # Export to CSV

# Comparisons
python mcp/server.py diff trace1.json trace2.json     # Compare kernels

# Deep Analysis
python mcp/server.py attribution trace.json           # Round breakdown
python mcp/server.py tail trace.json                  # Tail/drain analysis

# Code Analysis (static)
python mcp/server.py slot-packing                     # Instruction packing
python mcp/server.py gather                           # Memory patterns
python mcp/server.py hash                             # Hash efficiency
python mcp/server.py address                          # Register usage

# Auto-tuning
python mcp/server.py sweep --num-parallel 4,5,6,7,8   # Find best params
```

## Understanding the Output

### Key Metrics

**Cycle Count** - Your main target

- Current: Check your trace output
- Target: < 1500 cycles for excellence

**Utilization** - How well you use hardware

- Load: Should be >80% during gather phases
- Valu: Should be >70% during compute
- Low utilization = bottleneck!

**Bubble Cycles** - Wasted idle cycles

- Target: < 5% of total cycles
- High bubbles = poor pipelining

**Tail Drain** - Cycles after last gather

- Excellent: < 10 cycles
- Target for sub-1500: minimize this!

## Common Workflows

### Workflow 1: First Time Analysis

```bash
python mcp/server.py trace run --analyze
python mcp/server.py attribution trace.json
python mcp/server.py slot-packing
```

### Workflow 2: After Optimization

```bash
python mcp/server.py trace run --output trace_new.json
python mcp/server.py diff trace_old.json trace_new.json
```

### Workflow 3: Deep Dive

```bash
python mcp/server.py trace run --analyze --csv full_stats.csv
python mcp/server.py tail trace.json
python mcp/server.py hash
python mcp/server.py gather
python mcp/server.py address
```

## Tips

1. **Start with trace** - Always run `trace run --analyze` first
2. **Use diff** - Compare before/after every change
3. **Watch utilization** - Low utilization = your bottleneck
4. **Minimize tail** - Post-gather cycles kill performance
5. **Export CSV** - Use `--csv` for Excel/plotting

## Help

```bash
python mcp/server.py --help                    # Main help
python mcp/server.py trace --help              # Trace help
python mcp/server.py diff --help               # Diff help
```

## Full Documentation

See `mcp/README.md` for complete documentation.

---

**Ready to optimize? Start here:**

```bash
python mcp/server.py trace run --analyze
```

Good luck! 🚀
