# Performance MCP Tooling Suite

Comprehensive analysis and optimization tools for the Anthropic Performance Take-Home.

## 🎯 Overview

This MCP (Model Context Protocol) suite provides **9 specialized tools** organized into 3 tiers, designed to help you optimize your kernel from baseline (~147k cycles) down to sub-1500 cycles.

### Tool Tiers

**✅ Tier 1 - Must Have (Unlock Last 1000 Cycles)**

- 🥇 **Trace MCP**: Run kernel + analyze trace, compute per-cycle utilization
- 🥈 **Kernel Diff MCP**: Compare two versions with delta analysis
- 🥉 **Round Attribution MCP**: Tag cycles by round, group, tail

**✅ Tier 2 - High ROI (Make Tier-1 Stronger)**

- **Slot Packing MCP**: Find missed packing opportunities
- **Gather Pressure MCP**: Analyze memory access patterns
- **Tail/Drain MCP**: Measure post-gather and pre-store cycles

**⚙️ Tier 3 - Exploration & Auto-tuning**

- **Parameter Sweep MCP**: Brute-force optimal parameters
- **Hash Stage Cost MCP**: Identify algebraic fusion opportunities
- **Address Lifetime MCP**: Track register pressure

---

## 🚀 Quick Start

```bash
# Run kernel with trace and auto-analyze
python mcp/server.py trace run --analyze

# Compare two kernel versions
python mcp/server.py diff trace_old.json trace_new.json

# Full analysis suite
python mcp/server.py trace run --analyze --csv results.csv
python mcp/server.py attribution trace.json
python mcp/server.py tail trace.json
python mcp/server.py slot-packing
python mcp/server.py gather
```

---

## 📚 Tool Documentation

### 1. Trace MCP (Tier 1) 🥇

**Purpose**: Run kernel with trace + analyze utilization, bubble cycles, idle streaks.

**Mode A - Analyze Existing Trace**:

```bash
python mcp/server.py trace analyze trace.json --verbose
python mcp/server.py trace analyze trace.json --csv output.csv
```

**Mode B - Run + Trace + Auto-Analyze** (Best):

```bash
python mcp/server.py trace run --rounds 16 --batch 256 --analyze
python mcp/server.py trace run --output my_trace.json --analyze --csv stats.csv
```

**What It Shows**:

- Total cycles
- Load/Valu/ALU/Store utilization (%)
- Bubble cycles (idle)
- Longest idle streak
- Avg slots per instruction
- Tail drain cycles (after last gather)
- Per-cycle breakdown (first 20 + last 20)

**Use When**:

- You tweaked code and want quick post-mortem
- Need to identify which engine is starving
- Want to see if you have bubble cycles

---

### 2. Kernel Diff MCP (Tier 1) 🥈

**Purpose**: Compare two kernel versions with cycle deltas and utilization changes.

```bash
python mcp/server.py diff trace_v1.json trace_v2.json
python mcp/server.py diff trace_v1.json trace_v2.json --label-a "Baseline" --label-b "Optimized"
```

**What It Shows**:

- Cycle count delta (absolute + percentage)
- Engine utilization deltas
- Bubble cycle changes
- Tail drain changes
- Bottleneck analysis for both versions

**Use When**:

- You made a change and want to A/B test
- Need to see if optimization actually helped
- Want to understand what changed

---

### 3. Round Attribution MCP (Tier 1) 🥉

**Purpose**: Tag cycles by round, group, leftover, drain.

```bash
python mcp/server.py attribution trace.json
python mcp/server.py attribution trace.json --rounds 16 --vectors 32 --parallel 6
```

**What It Shows**:

- Cycles per round (round-by-round breakdown)
- Average cycles for early rounds (0-2, no gather)
- Average cycles for late rounds (3+, with gather)
- Tail/drain cycles
- Top 5 slowest rounds

**Use When**:

- You want to know where time is going
- Need to find which rounds are slow
- Want to verify round 3+ has proper overlap

---

### 4. Slot Packing MCP (Tier 2)

**Purpose**: Scan instruction stream for packing efficiency.

```bash
python mcp/server.py slot-packing
python mcp/server.py slot-packing --rounds 16 --batch 256
```

**What It Shows**:

- Average slot usage per engine
- Overall packing efficiency (%)
- Worst 5% bundles (lowest efficiency)
- Longest single-engine runs (load-only, valu-only)
- Recommendations for better packing

**Use When**:

- Before running trace (static analysis)
- Want to find missed packing opportunities
- Need to see if you have long single-engine runs

---

### 5. Gather Pressure MCP (Tier 2)

**Purpose**: Analyze memory access patterns.

```bash
python mcp/server.py gather
```

**What It Shows**:

- Scalar loads (gather operations)
- Vector loads
- Addresses loaded multiple times
- Address computation patterns
- Address lifetimes (compute → use gap)
- Repeated address computations

**Use When**:

- You suspect you're burning memory bandwidth
- Want to find repeated loads
- Need to understand gather pressure

---

### 6. Tail/Drain MCP (Tier 2)

**Purpose**: Measure post-gather and pre-store cycles.

```bash
python mcp/server.py tail trace.json
```

**What It Shows**:

- Cycles after last gather
- Cycles after last compute
- Store phase length
- Pure store cycles vs overlapped
- Store overlap efficiency
- Tail cycles breakdown

**Use When**:

- You want sub-1500 cycles (1363-cycle kernels have minimal tail)
- Need to see if stores are overlapped with compute
- Want to understand drain phase

---

### 7. Parameter Sweep MCP (Tier 3) ⚙️

**Purpose**: Auto-rerun kernel with different parameters.

```bash
python mcp/server.py sweep --num-parallel 4,5,6,7,8
python mcp/server.py sweep --json sweep_results.json
```

**What It Shows**:

- Cycle count for each parameter combination
- Top 10 best configurations
- Best vs worst comparison
- Improvement percentage

**Use When**:

- You don't know optimal NUM_PARALLEL
- Want to brute-force parameter space
- Testing different strategies (mux vs gather)

**Note**: Requires kernel to be parameterizable.

---

### 8. Hash Stage Cost MCP (Tier 3) ⚙️

**Purpose**: Analyze hash computation efficiency.

```bash
python mcp/server.py hash
```

**What It Shows**:

- Hash operation counts (XOR, multiply_add)
- Collapsed vs non-collapsed stages
- Cycle cost estimation
- Collapsible stages (can use multiply_add)
- Hash/Load overlap ratio

**Use When**:

- Want to algebraically fuse hash operations
- Need to see if hash stages are collapsed
- Want to verify hash is overlapped with gather

---

### 9. Address Lifetime MCP (Tier 3) ⚙️

**Purpose**: Track address register lifetimes.

```bash
python mcp/server.py address
```

**What It Shows**:

- Total address registers
- Average lifetime (instrs)
- Long-lived addresses (>50 instrs)
- Lifetime distribution
- Top longest-lived addresses with details
- Recommendations for register reuse

**Use When**:

- You suspect register pressure
- Want to optimize address computation timing
- Need to find over-buffering (long addr lifetimes)

---

## 🔬 Practical Workflows

### Workflow 1: Initial Diagnosis

```bash
# Run and get baseline
python mcp/server.py trace run --analyze

# Identify bottlenecks
python mcp/server.py slot-packing
python mcp/server.py gather
```

### Workflow 2: After Making Changes

```bash
# Run new trace
python mcp/server.py trace run --output trace_v2.json

# Compare
python mcp/server.py diff trace_v1.json trace_v2.json
```

### Workflow 3: Deep Dive on Performance

```bash
# Full suite
python mcp/server.py trace run --analyze --csv stats.csv
python mcp/server.py attribution trace.json
python mcp/server.py tail trace.json
python mcp/server.py hash
python mcp/server.py address
```

### Workflow 4: Parameter Tuning

```bash
# Sweep parameters
python mcp/server.py sweep --num-parallel 4,5,6,7,8 --json results.json

# Analyze best result
# (update kernel with best params, then re-trace)
```

---

## 📊 Understanding Output

### Key Metrics to Watch

**Cycle Count**: Your main optimization target

- Baseline: ~147k cycles
- Good: <3000 cycles
- Excellent: <1500 cycles
- World-class: <1400 cycles

**Load Utilization**: Should be high in rounds 3+

- Target: >80% during gather phases
- Red flag: <50% (not gathering enough)

**Valu Utilization**: Hash computation

- Target: >70% during compute phases
- Red flag: <40% (not packing hash ops)

**Bubble Cycles**: Completely idle cycles

- Target: <5% of total
- Red flag: >10% (poor pipelining)

**Tail Drain**: Cycles after last gather

- Excellent: <10 cycles
- Good: 10-50 cycles
- Poor: >100 cycles

**Store Overlap**: Stores interleaved with compute

- Excellent: >70%
- Poor: <30% (stores happening alone)

---

## 🎯 Optimization Strategy

### Phase 1: Understand Current State

1. Run `trace run --analyze` to get baseline
2. Run `attribution` to see round breakdown
3. Run `slot-packing` to see packing efficiency

### Phase 2: Identify Bottleneck

- Low load util? → Not gathering enough data
- Low valu util? → Not packing hash ops well
- High bubble cycles? → Poor pipelining
- High tail drain? → Not overlapping store with compute

### Phase 3: Fix and Verify

1. Make code changes
2. Run `trace run --output trace_v2.json`
3. Run `diff trace_v1.json trace_v2.json`
4. Iterate!

### Phase 4: Polish (Sub-1500 cycles)

- Run `tail` to minimize drain
- Run `hash` to ensure collapsed stages
- Run `address` to reduce register pressure
- Run `sweep` to find optimal parameters

---

## 🔧 Advanced Features

### Export to CSV

```bash
python mcp/server.py trace run --analyze --csv trace_stats.csv
```

### Custom Labels for Diff

```bash
python mcp/server.py diff old.json new.json --label-a "Baseline" --label-b "Pipelined v2"
```

### Specify Forest Parameters

```bash
python mcp/server.py trace run --height 10 --rounds 16 --batch 256
```

---

## 📖 Example Session

```bash
$ python mcp/server.py trace run --analyze
Running kernel: forest_height=10, rounds=16, batch_size=256
✓ Kernel completed successfully
  Cycles: 2489
  Trace saved to: trace.json

======================================================================
TRACE ANALYSIS: trace.json
======================================================================

Total Cycles: 2489

Engine Utilization:
  Load:   65.32% (avg 1.31/2 slots/cycle)
  Valu:   78.45% (avg 4.71/6 slots/cycle)
  ALU:    12.34% (avg 1.48/12 slots/cycle)
  Store:  2.10% (avg 0.04/2 slots/cycle)

Bubble Cycles: 23
Longest Idle Streak: 5
Avg Slots per Instruction: 7.54
Tail Drain Cycles: 42

$ python mcp/server.py diff trace_v1.json trace_v2.json

======================================================================
KERNEL COMPARISON: Version A vs Version B
======================================================================

┌────────────────────────────────────────────────────────────────────┐
│                           CYCLE COUNT                              │
├────────────────────────────────────────────────────────────────────┤
│ Version A                     :     2489 cycles                    │
│ Version B                     :     2311 cycles                    │
├────────────────────────────────────────────────────────────────────┤
│ Delta:                         -178 cycles (-7.15%)                │
│                               >>> FASTER                           │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

**Q: "No module named 'trace_mcp'"**
A: Run from project root: `python mcp/server.py ...`

**Q: Trace file not found**
A: Ensure you run `trace run` before `trace analyze`

**Q: Parameter sweep doesn't change results**
A: Kernel needs to be modified to accept parameters (template provided)

---

## 🚀 Performance Benchmarks

Using these tools, you can target:

- **2164 cycles**: Claude Opus 4 (many hours)
- **1790 cycles**: Human expert (2 hours)
- **1579 cycles**: Claude Opus 4.5 (2 hours)
- **1487 cycles**: Claude Opus 4.5 (11.5 hours)
- **1363 cycles**: Claude Opus 4.5 (improved harness)
- **???** cycles: Best human ever (undisclosed)

Good luck optimizing! 🎉

---

## 📝 License

Part of Anthropic's Performance Take-Home. For educational purposes.
