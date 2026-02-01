"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.current_bundle = {k: [] for k in SLOT_LIMITS}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def flush(self):
        """Emit the current bundle if non-empty"""
        bundle = {k: v for k, v in self.current_bundle.items() if v}
        if bundle:
            self.instrs.append(bundle)
        self.current_bundle = {k: [] for k in SLOT_LIMITS}
    
    def pack(self, engine, slot):
        """Add to current bundle if space, else flush and add"""
        if len(self.current_bundle[engine]) >= SLOT_LIMITS[engine]:
            self.flush()
        self.current_bundle[engine].append(slot)

    def add(self, engine, slot):
        """Legacy: emit single-op instruction"""
        self.flush()
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def alloc_const_batch(self, const_list):
        """Pre-allocate and emit constants in batches"""
        addrs = []
        for val, name in const_list:
            if val in self.const_map:
                addrs.append(self.const_map[val])
            else:
                addr = self.alloc_scratch(name)
                self.const_map[val] = addr
                addrs.append(addr)
        
        # Emit all new constants in batches of 2
        pending = [(addr, val) for (val, name), addr in zip(const_list, addrs) 
                   if val not in self.const_map or self.const_map[val] == addr]
        for i in range(0, len(pending), 2):
            batch = pending[i:i+2]
            ops = [("const", addr, val) for addr, val in batch]
            self.instrs.append({"load": ops})
        return addrs

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized SIMD implementation.
        Key optimizations:
        1. Process VLEN=8 elements per cycle instead of 1
        2. Keep batch in scratch registers to avoid memory round-trips
        3. Use multiply_add for fused hash operations
        4. Pack multiple ops per bundle using VLIW
        """
        n_vectors = batch_size // VLEN  # 32 vectors of 8 elements
        
        # === ALLOCATE SCRATCH ===
        tmp = self.alloc_scratch("tmp")
        tmp2 = self.alloc_scratch("tmp2")
        
        # Load initial variables from memory addresses 0-6
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp, i))
            self.add("load", ("load", self.scratch[v], tmp))
        
        # Scalar constants
        zero_s = self.scratch_const(0, "zero_s")
        one_s = self.scratch_const(1, "one_s")
        two_s = self.scratch_const(2, "two_s")
        
        # Vector constants (broadcast from scalars)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        self.add("valu", ("vbroadcast", v_zero, zero_s))
        self.add("valu", ("vbroadcast", v_one, one_s))
        self.add("valu", ("vbroadcast", v_two, two_s))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        
        # Hash constants (scalar and vector)
        hash_scalar = []
        hash_vector = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c_s = self.scratch_const(val1, f"hc_{hi}")
            c_v = self.alloc_scratch(f"hc_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c_v, c_s))
            
            # Aux: either shift amount or fused multiplier
            if op1 == "+" and op2 == "+" and op3 == "<<":
                aux_val = 1 + (1 << val3)
            else:
                aux_val = val3
            aux_s = self.scratch_const(aux_val, f"haux_{hi}")
            aux_v = self.alloc_scratch(f"haux_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", aux_v, aux_s))
            
            hash_scalar.append((c_s, aux_s, op1, op2, op3, val3))
            hash_vector.append((c_v, aux_v))
        
        # Batch vectors - keep in scratch the entire time
        indices = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(n_vectors)]
        values = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(n_vectors)]
        
        # Temporaries
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_node = self.alloc_scratch("v_node", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)
        
        # === LOAD ENTIRE BATCH INTO SCRATCH ===
        for vi in range(n_vectors):
            offset_s = self.scratch_const(vi * VLEN, f"off_{vi}")
            self.add("alu", ("+", tmp, self.scratch["inp_indices_p"], offset_s))
            self.add("load", ("vload", indices[vi], tmp))
            self.add("alu", ("+", tmp, self.scratch["inp_values_p"], offset_s))
            self.add("load", ("vload", values[vi], tmp))
        
        self.add("flow", ("pause",))
        
        # === MAIN LOOP ===
        for r in range(rounds):
            for vi in range(n_vectors):
                v_idx = indices[vi]
                v_val = values[vi]
                
                # --- GATHER: load node values from forest[idx] ---
                # This is scatter-gather: each lane has different index
                # Must broadcast forest_ptr, add per-lane idx, then scalar loads
                self.add("valu", ("vbroadcast", v_addr, self.scratch["forest_values_p"]))
                self.add("valu", ("+", v_addr, v_addr, v_idx))
                
                # Scalar loads for gather (2 loads per cycle)
                for lane in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", v_node + lane, v_addr + lane),
                        ("load", v_node + lane + 1, v_addr + lane + 1)
                    ]})
                
                # --- XOR with node values ---
                self.add("valu", ("^", v_val, v_val, v_node))
                
                # --- HASH (fused where possible) ---
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c_v, aux_v = hash_vector[hi]
                    
                    if op1 == "+" and op2 == "+" and op3 == "<<":
                        # Fused: val = val * (1 + 2^shift) + c
                        self.add("valu", ("multiply_add", v_val, v_val, aux_v, c_v))
                    else:
                        # Two ops then combine
                        self.add("valu", (op1, v_tmp1, v_val, c_v))
                        self.add("valu", (op3, v_tmp2, v_val, aux_v))
                        self.add("valu", (op2, v_val, v_tmp1, v_tmp2))
                
                # --- INDEX UPDATE: idx = 2*idx + 1 + (val & 1) ---
                # Equivalent: idx = idx * 2 + (1 + (val & 1))
                self.add("valu", ("&", v_tmp1, v_val, v_one))
                self.add("valu", ("+", v_tmp1, v_tmp1, v_one))
                self.add("valu", ("multiply_add", v_idx, v_idx, v_two, v_tmp1))
                
                # --- WRAP: idx = idx * (idx < n_nodes) ---
                self.add("valu", ("<", v_tmp1, v_idx, v_n_nodes))
                self.add("valu", ("*", v_idx, v_idx, v_tmp1))
        
        self.add("flow", ("pause",))
        
        # === STORE RESULTS BACK TO MEMORY ===
        for vi in range(n_vectors):
            offset_s = self.scratch_const(vi * VLEN)
            self.add("alu", ("+", tmp, self.scratch["inp_indices_p"], offset_s))
            self.add("store", ("vstore", tmp, indices[vi]))
            self.add("alu", ("+", tmp, self.scratch["inp_values_p"], offset_s))
            self.add("store", ("vstore", tmp, values[vi]))

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
