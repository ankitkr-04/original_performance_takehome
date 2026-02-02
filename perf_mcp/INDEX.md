# MCP Tooling Suite Index

## 📚 Documentation Files

- **[QUICKSTART.md](QUICKSTART.md)** - Start here! Quick guide to get running in 5 minutes
- **[README.md](README.md)** - Complete documentation with all details
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details

## 🛠️ Tool Files

### Main Entry Point

- **[server.py](server.py)** - Unified CLI interface for all tools

### Tier 1 Tools (Must Have)

- **[trace_mcp.py](trace_mcp.py)** - Run and analyze kernel traces
- **[kernel_diff_mcp.py](kernel_diff_mcp.py)** - Compare two kernel versions
- **[round_attribution_mcp.py](round_attribution_mcp.py)** - Attribute cycles to rounds

### Tier 2 Tools (High ROI)

- **[slot_packing_mcp.py](slot_packing_mcp.py)** - Analyze instruction packing efficiency
- **[gather_pressure_mcp.py](gather_pressure_mcp.py)** - Analyze memory access patterns
- **[tail_drain_mcp.py](tail_drain_mcp.py)** - Analyze tail and drain phases

### Tier 3 Tools (Exploration)

- **[param_sweep_mcp.py](param_sweep_mcp.py)** - Auto-tune parameters
- **[hash_stage_mcp.py](hash_stage_mcp.py)** - Analyze hash computation
- **[address_lifetime_mcp.py](address_lifetime_mcp.py)** - Analyze register usage

## 🚀 Quick Commands

```bash
# Get help
python mcp/server.py --help

# Run trace and analyze
python mcp/server.py trace run --analyze

# Static analysis (no trace needed)
python mcp/server.py slot-packing
python mcp/server.py gather
python mcp/server.py hash
python mcp/server.py address

# Trace-based analysis
python mcp/server.py trace analyze trace.json
python mcp/server.py diff trace1.json trace2.json
python mcp/server.py attribution trace.json
python mcp/server.py tail trace.json

# Auto-tuning
python mcp/server.py sweep --num-parallel 4,5,6,7
```

## 📖 Recommended Reading Order

1. **First Time?** Read [QUICKSTART.md](QUICKSTART.md)
2. **Want Details?** Read [README.md](README.md)
3. **Technical Deep Dive?** Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 🎯 Workflow Recommendations

### For Beginners

```bash
# 1. Understand current performance
python mcp/server.py trace run --analyze
python mcp/server.py slot-packing

# 2. Make a change, compare
python mcp/server.py trace run --output trace_v2.json
python mcp/server.py diff trace.json trace_v2.json
```

### For Intermediate

```bash
# Full analysis suite
python mcp/server.py trace run --analyze --csv stats.csv
python mcp/server.py attribution trace.json
python mcp/server.py tail trace.json
python mcp/server.py gather
python mcp/server.py hash
```

### For Advanced

```bash
# Deep optimization
python mcp/server.py address           # Check register pressure
python mcp/server.py sweep             # Find optimal params
python mcp/server.py tail trace.json   # Minimize tail cycles
```

## 🏆 Performance Targets

| Target        | Cycles    | Requirement           |
| ------------- | --------- | --------------------- |
| Pass baseline | < 147,734 | Beat starter code     |
| Good          | < 3,000   | Solid optimization    |
| Excellent     | < 1,500   | World-class           |
| Elite         | < 1,400   | Beat best Claude      |
| Legend        | < 1,363   | Match Claude Opus 4.5 |

## 💡 Key Insights from Tools

- **High bubble cycles?** → Poor pipelining (check slot-packing)
- **Low load utilization?** → Not gathering enough (check gather)
- **Long tail drain?** → Overlap stores with compute (check tail)
- **Low valu utilization?** → Pack hash ops better (check hash)
- **Many long-lived addresses?** → Register pressure (check address)

## 🐛 Troubleshooting

**Problem**: "No module named 'trace_mcp'"

- **Solution**: Run from project root: `cd /path/to/project && python mcp/server.py ...`

**Problem**: "trace.json not found"

- **Solution**: Generate it first: `python mcp/server.py trace run`

**Problem**: "SyntaxError in server.py"

- **Solution**: File is fixed, pull latest version

## 📞 Getting Help

```bash
# Main help
python mcp/server.py --help

# Tool-specific help
python mcp/server.py trace --help
python mcp/server.py diff --help
python mcp/server.py [tool] --help
```

## 🎉 You're Ready!

Pick your starting point:

- 👉 **New?** → [QUICKSTART.md](QUICKSTART.md)
- 👉 **Optimizing?** → `python mcp/server.py trace run --analyze`
- 👉 **Learning?** → [README.md](README.md)

Good luck reaching sub-1500 cycles! 🚀
