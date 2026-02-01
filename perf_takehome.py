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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel with 8-way parallel processing.
        
        Key optimizations:
        1. Process 8 vectors at a time (maximize 6 VALU slots utilization)
        2. Better gather interleaving
        3. Fused multiply_add for hash stages
        """
        n_vectors = batch_size // VLEN  # 32 vectors
        NUM_PARALLEL = 8  # Process more vectors in parallel
        
        # === SCRATCH ALLOCATION ===
        tmp = self.alloc("tmp")
        tmp2 = self.alloc("tmp2")
        
        # Pointers from memory
        forest_p = self.alloc("forest_p")
        idx_p = self.alloc("idx_p")
        val_p = self.alloc("val_p")
        n_nodes_s = self.alloc("n_nodes_s")
        
        # Scalar constants
        zero_s = self.alloc("zero_s")
        one_s = self.alloc("one_s")
        two_s = self.alloc("two_s")
        
        # Offsets for batch loading
        offsets = [self.alloc(f"off_{vi}") for vi in range(n_vectors)]
        
        # Vector constants
        v_one = self.alloc("v_one", VLEN)
        v_two = self.alloc("v_two", VLEN)
        v_n_nodes = self.alloc("v_n_nodes", VLEN)
        
        # Hash constants (scalar then vector)
        hash_s_c = []
        hash_s_aux = []
        hash_v_c = []
        hash_v_aux = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_s_c.append(self.alloc(f"hsc_{hi}"))
            hash_v_c.append(self.alloc(f"hvc_{hi}", VLEN))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                aux_val = 1 + (1 << val3)
            else:
                aux_val = val3
            hash_s_aux.append((self.alloc(f"hsa_{hi}"), aux_val))
            hash_v_aux.append(self.alloc(f"hva_{hi}", VLEN))
        
        # Batch vectors
        indices = [self.alloc(f"idx_{i}", VLEN) for i in range(n_vectors)]
        values = [self.alloc(f"val_{i}", VLEN) for i in range(n_vectors)]
        
        # Working registers for parallel processing
        v_node = [self.alloc(f"v_node_{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_addr = [self.alloc(f"v_addr_{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_tmp1 = [self.alloc(f"v_tmp1_{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_tmp2 = [self.alloc(f"v_tmp2_{i}", VLEN) for i in range(NUM_PARALLEL)]
        
        # === INITIALIZATION ===
        self.instrs.append({"load": [("const", tmp, 1), ("const", tmp2, 4)]})
        self.instrs.append({"load": [("load", n_nodes_s, tmp), ("load", forest_p, tmp2)]})
        self.instrs.append({"load": [("const", tmp, 5), ("const", tmp2, 6)]})
        self.instrs.append({"load": [("load", idx_p, tmp), ("load", val_p, tmp2)]})
        
        self.instrs.append({"load": [("const", zero_s, 0), ("const", one_s, 1)]})
        self.instrs.append({"load": [("const", two_s, 2)]})
        
        for vi in range(0, n_vectors, 2):
            ops = [("const", offsets[vi], vi * VLEN)]
            if vi + 1 < n_vectors:
                ops.append(("const", offsets[vi + 1], (vi + 1) * VLEN))
            self.instrs.append({"load": ops})
        
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            s_aux, aux_val = hash_s_aux[hi]
            self.instrs.append({"load": [("const", hash_s_c[hi], val1), ("const", s_aux, aux_val)]})
        
        self.instrs.append({"valu": [
            ("vbroadcast", v_one, one_s),
            ("vbroadcast", v_two, two_s),
            ("vbroadcast", v_n_nodes, n_nodes_s)
        ]})
        
        for hi in range(0, len(HASH_STAGES), 3):
            ops = []
            for hj in range(3):
                if hi + hj < len(HASH_STAGES):
                    s_aux, _ = hash_s_aux[hi + hj]
                    ops.append(("vbroadcast", hash_v_c[hi + hj], hash_s_c[hi + hj]))
                    ops.append(("vbroadcast", hash_v_aux[hi + hj], s_aux))
            self.instrs.append({"valu": ops})
        
        for vi in range(n_vectors):
            self.instrs.append({"alu": [
                ("+", tmp, idx_p, offsets[vi]),
                ("+", tmp2, val_p, offsets[vi])
            ]})
            self.instrs.append({"load": [
                ("vload", indices[vi], tmp),
                ("vload", values[vi], tmp2)
            ]})
        
        self.instrs.append({"flow": [("pause",)]})
        
        # === MAIN LOOP: Process 8 vectors at a time ===
        for r in range(rounds):
            for vi_base in range(0, n_vectors, NUM_PARALLEL):
                vi_list = list(range(vi_base, min(vi_base + NUM_PARALLEL, n_vectors)))
                n = len(vi_list)
                
                # Broadcast forest pointer to all address vectors (6 valu slots)
                for start in range(0, n, 6):
                    ops = [("vbroadcast", v_addr[p], forest_p) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
                
                # Add indices to addresses (6 valu slots)
                for start in range(0, n, 6):
                    ops = [("+", v_addr[p], v_addr[p], indices[vi_list[p]]) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
                
                # GATHER: 2 loads per cycle
                # Interleave across all 8 vectors for better pipelining
                for lane in range(VLEN):
                    for p_start in range(0, n, 2):
                        ops = []
                        for p in range(p_start, min(p_start + 2, n)):
                            ops.append(("load", v_node[p] + lane, v_addr[p] + lane))
                        self.instrs.append({"load": ops})
                
                # XOR with node values (6 valu slots)
                for start in range(0, n, 6):
                    ops = [("^", values[vi_list[p]], values[vi_list[p]], v_node[p]) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
                
                # HASH (6 stages)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c_v = hash_v_c[hi]
                    aux_v = hash_v_aux[hi]
                    
                    if op1 == "+" and op2 == "+" and op3 == "<<":
                        # Fused multiply-add (6 valu slots)
                        for start in range(0, n, 6):
                            ops = [("multiply_add", values[vi_list[p]], values[vi_list[p]], aux_v, c_v) for p in range(start, min(start + 6, n))]
                            self.instrs.append({"valu": ops})
                    else:
                        # Standard: compute both operands then combine
                        for start in range(0, n, 3):
                            ops = []
                            for p in range(start, min(start + 3, n)):
                                ops.append((op1, v_tmp1[p], values[vi_list[p]], c_v))
                                ops.append((op3, v_tmp2[p], values[vi_list[p]], aux_v))
                            self.instrs.append({"valu": ops})
                        
                        for start in range(0, n, 6):
                            ops = [(op2, values[vi_list[p]], v_tmp1[p], v_tmp2[p]) for p in range(start, min(start + 6, n))]
                            self.instrs.append({"valu": ops})
                
                # INDEX UPDATE: idx = 2*idx + (1 + (val&1))
                for start in range(0, n, 6):
                    ops = [("&", v_tmp1[p], values[vi_list[p]], v_one) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
                
                for start in range(0, n, 6):
                    ops = [("+", v_tmp1[p], v_tmp1[p], v_one) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
                
                for start in range(0, n, 6):
                    ops = [("multiply_add", indices[vi_list[p]], indices[vi_list[p]], v_two, v_tmp1[p]) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
                
                # WRAP: idx = idx * (idx < n_nodes)
                for start in range(0, n, 6):
                    ops = [("<", v_tmp1[p], indices[vi_list[p]], v_n_nodes) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
                
                for start in range(0, n, 6):
                    ops = [("*", indices[vi_list[p]], indices[vi_list[p]], v_tmp1[p]) for p in range(start, min(start + 6, n))]
                    self.instrs.append({"valu": ops})
        
        self.instrs.append({"flow": [("pause",)]})
        
        # Store results
        for vi in range(n_vectors):
            self.instrs.append({"alu": [
                ("+", tmp, idx_p, offsets[vi]),
                ("+", tmp2, val_p, offsets[vi])
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

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
