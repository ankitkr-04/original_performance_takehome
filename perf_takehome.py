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
        Heavily optimized vectorized SIMD implementation.
        Key optimizations:
        1. Process VLEN=8 elements per cycle instead of 1
        2. Keep batch in scratch registers to avoid memory round-trips
        3. Use multiply_add for fused hash operations
        4. Pack multiple VALU ops per cycle (6 slots available)
        5. Process 2 vectors in parallel to better utilize VALU slots
        """
        n_vectors = batch_size // VLEN  # 32 vectors of 8 elements
        
        # === ALLOCATE ALL SCRATCH FIRST ===
        tmp = self.alloc_scratch("tmp")
        tmp2 = self.alloc_scratch("tmp2")
        
        # Allocate init variables
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        # Pre-allocate all constants we'll need  
        zero_s = self.alloc_scratch("zero_s")
        one_s = self.alloc_scratch("one_s")
        two_s = self.alloc_scratch("two_s")
        
        # Offset constants for batch loading
        offsets = [self.alloc_scratch(f"off_{vi}") for vi in range(n_vectors)]
        
        # Vector constants
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        # Hash constants (both scalar and vector)
        hash_v_c = []
        hash_v_aux = []
        hash_s_c = []
        hash_s_aux = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_s_c.append(self.alloc_scratch(f"hsc_{hi}"))
            hash_v_c.append(self.alloc_scratch(f"hvc_{hi}", VLEN))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                aux_val = 1 + (1 << val3)
            else:
                aux_val = val3
            hash_s_aux.append((self.alloc_scratch(f"hsa_{hi}"), aux_val))
            hash_v_aux.append(self.alloc_scratch(f"hva_{hi}", VLEN))
        
        # Batch vectors - keep in scratch the entire time
        indices = [self.alloc_scratch(f"idx_{i}", VLEN) for i in range(n_vectors)]
        values = [self.alloc_scratch(f"val_{i}", VLEN) for i in range(n_vectors)]
        
        # Temporaries for 2-way parallel processing
        v_tmp1_a = self.alloc_scratch("v_tmp1_a", VLEN)
        v_tmp2_a = self.alloc_scratch("v_tmp2_a", VLEN)
        v_node_a = self.alloc_scratch("v_node_a", VLEN)
        v_addr_a = self.alloc_scratch("v_addr_a", VLEN)
        v_tmp1_b = self.alloc_scratch("v_tmp1_b", VLEN)
        v_tmp2_b = self.alloc_scratch("v_tmp2_b", VLEN)
        v_node_b = self.alloc_scratch("v_node_b", VLEN)
        v_addr_b = self.alloc_scratch("v_addr_b", VLEN)
        
        # === EMIT INITIALIZATION ===
        # Load init variables from memory
        for i, v in enumerate(init_vars):
            self.instrs.append({"load": [("const", tmp, i)]})
            self.instrs.append({"load": [("load", self.scratch[v], tmp)]})
        
        # Load scalar constants (packed)
        self.instrs.append({"load": [("const", zero_s, 0), ("const", one_s, 1)]})
        self.instrs.append({"load": [("const", two_s, 2)]})
        
        # Load offset constants (packed - 2 per cycle)
        for vi in range(0, n_vectors, 2):
            ops = [("const", offsets[vi], vi * VLEN)]
            if vi + 1 < n_vectors:
                ops.append(("const", offsets[vi + 1], (vi + 1) * VLEN))
            self.instrs.append({"load": ops})
        
        # Load hash scalar constants (packed)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            s_aux, aux_val = hash_s_aux[hi]
            self.instrs.append({"load": [("const", hash_s_c[hi], val1), ("const", s_aux, aux_val)]})
        
        # Broadcast vector constants (packed - 6 VALU slots)
        self.instrs.append({"valu": [
            ("vbroadcast", v_one, one_s),
            ("vbroadcast", v_two, two_s),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])
        ]})
        
        # Broadcast hash constants (3 per cycle = 6 slots)
        for hi in range(0, len(HASH_STAGES), 3):
            ops = []
            for hj in range(3):
                if hi + hj < len(HASH_STAGES):
                    s_aux, _ = hash_s_aux[hi + hj]
                    ops.append(("vbroadcast", hash_v_c[hi + hj], hash_s_c[hi + hj]))
                    ops.append(("vbroadcast", hash_v_aux[hi + hj], s_aux))
            self.instrs.append({"valu": ops})
        
        # === LOAD ENTIRE BATCH INTO SCRATCH (packed vload) ===
        for vi in range(n_vectors):
            self.instrs.append({"alu": [
                ("+", tmp, self.scratch["inp_indices_p"], offsets[vi]),
                ("+", tmp2, self.scratch["inp_values_p"], offsets[vi])
            ]})
            self.instrs.append({"load": [
                ("vload", indices[vi], tmp),
                ("vload", values[vi], tmp2)
            ]})
        
        self.instrs.append({"flow": [("pause",)]})
        
        # === MAIN LOOP: Process 2 vectors at a time ===
        for r in range(rounds):
            for vi in range(0, n_vectors, 2):
                idx_a, val_a = indices[vi], values[vi]
                idx_b, val_b = indices[vi + 1], values[vi + 1]
                
                # --- Compute addresses for both vectors (parallel) ---
                self.instrs.append({"valu": [
                    ("vbroadcast", v_addr_a, self.scratch["forest_values_p"]),
                    ("vbroadcast", v_addr_b, self.scratch["forest_values_p"])
                ]})
                self.instrs.append({"valu": [
                    ("+", v_addr_a, v_addr_a, idx_a),
                    ("+", v_addr_b, v_addr_b, idx_b)
                ]})
                
                # --- Gather for vector A (4 cycles) ---
                for lane in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", v_node_a + lane, v_addr_a + lane),
                        ("load", v_node_a + lane + 1, v_addr_a + lane + 1)
                    ]})
                
                # --- Gather for vector B (4 cycles) ---
                for lane in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", v_node_b + lane, v_addr_b + lane),
                        ("load", v_node_b + lane + 1, v_addr_b + lane + 1)
                    ]})
                
                # --- XOR both vectors (parallel) ---
                self.instrs.append({"valu": [
                    ("^", val_a, val_a, v_node_a),
                    ("^", val_b, val_b, v_node_b)
                ]})
                
                # --- HASH both vectors in parallel ---
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c_v = hash_v_c[hi]
                    aux_v = hash_v_aux[hi]
                    
                    if op1 == "+" and op2 == "+" and op3 == "<<":
                        # Fused: val = val * mult + c (2 ops, fits in 6 slots)
                        self.instrs.append({"valu": [
                            ("multiply_add", val_a, val_a, aux_v, c_v),
                            ("multiply_add", val_b, val_b, aux_v, c_v)
                        ]})
                    else:
                        # 4 parallel ops, then 2 combine ops
                        self.instrs.append({"valu": [
                            (op1, v_tmp1_a, val_a, c_v),
                            (op3, v_tmp2_a, val_a, aux_v),
                            (op1, v_tmp1_b, val_b, c_v),
                            (op3, v_tmp2_b, val_b, aux_v)
                        ]})
                        self.instrs.append({"valu": [
                            (op2, val_a, v_tmp1_a, v_tmp2_a),
                            (op2, val_b, v_tmp1_b, v_tmp2_b)
                        ]})
                
                # --- INDEX UPDATE for both vectors ---
                # (val & 1) + 1 for both, then multiply_add
                self.instrs.append({"valu": [
                    ("&", v_tmp1_a, val_a, v_one),
                    ("&", v_tmp1_b, val_b, v_one)
                ]})
                self.instrs.append({"valu": [
                    ("+", v_tmp1_a, v_tmp1_a, v_one),
                    ("+", v_tmp1_b, v_tmp1_b, v_one)
                ]})
                self.instrs.append({"valu": [
                    ("multiply_add", idx_a, idx_a, v_two, v_tmp1_a),
                    ("multiply_add", idx_b, idx_b, v_two, v_tmp1_b)
                ]})
                
                # --- WRAP both indices ---
                self.instrs.append({"valu": [
                    ("<", v_tmp1_a, idx_a, v_n_nodes),
                    ("<", v_tmp1_b, idx_b, v_n_nodes)
                ]})
                self.instrs.append({"valu": [
                    ("*", idx_a, idx_a, v_tmp1_a),
                    ("*", idx_b, idx_b, v_tmp1_b)
                ]})
        
        self.instrs.append({"flow": [("pause",)]})
        
        # === STORE RESULTS BACK TO MEMORY (packed) ===
        for vi in range(n_vectors):
            self.instrs.append({"alu": [
                ("+", tmp, self.scratch["inp_indices_p"], offsets[vi]),
                ("+", tmp2, self.scratch["inp_values_p"], offsets[vi])
            ]})
            self.instrs.append({"store": [
                ("vstore", tmp, indices[vi]),
                ("vstore", tmp2, values[vi])
            ]})

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
