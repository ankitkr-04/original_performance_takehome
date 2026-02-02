"""
Optimized kernel targeting 1300 cycles.

Key insight: Looking at the trace analysis more carefully:
- Current impl gets 68% load util and 69% valu util
- But rounds 0-2 are valu-only (120 cycles each)
- Rounds 3+ are 180 cycles with some overlap

To improve:
1. Better cross-round pipelining: start gathering for round R while hashing round R-1
2. More aggressive interleaving within rounds
3. Pre-compute addresses earlier
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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr} > {SCRATCH_SIZE}"
        return addr

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel with cross-round pipelining.
        
        Groups of 6 vectors. For rounds 3+:
        - While hashing group G of round R, gather group G+1 of round R
        - At round boundary: flush last group hash, then start next round
        
        Better: at end of round R, start gathering group 0 of round R+1
        while finishing the hash of last group of round R.
        """
        n_vectors = batch_size // VLEN  # 32
        NUM_PARALLEL = 6
        n_groups = (n_vectors + NUM_PARALLEL - 1) // NUM_PARALLEL  # 6 groups
        
        def get_vi_list(gi):
            start = gi * NUM_PARALLEL
            end = min(start + NUM_PARALLEL, n_vectors)
            return list(range(start, end))
        
        # === SCRATCH ALLOCATION ===
        tmp = [self.alloc(f"tmp{i}") for i in range(16)]
        
        forest_p = self.alloc("forest_p")
        idx_p = self.alloc("idx_p")
        val_p = self.alloc("val_p")
        n_nodes_s = self.alloc("n_nodes_s")
        one_s = self.alloc("one_s")
        two_s = self.alloc("two_s")
        
        v_one = self.alloc("v_one", VLEN)
        v_two = self.alloc("v_two", VLEN)
        v_three = self.alloc("v_three", VLEN)
        v_n_nodes = self.alloc("v_n_nodes", VLEN)
        
        # Hash constants
        hash_v_c = []
        hash_v_aux = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_v_c.append(self.alloc(f"hvc_{hi}", VLEN))
            aux_val = (1 + (1 << val3)) if (op1 == "+" and op2 == "+" and op3 == "<<") else val3
            hash_v_aux.append((self.alloc(f"hva_{hi}", VLEN), aux_val))
        
        # Per-vector data
        indices = [self.alloc(f"idx_{i}", VLEN) for i in range(n_vectors)]
        values = [self.alloc(f"val_{i}", VLEN) for i in range(n_vectors)]
        v_idx_base = [self.alloc(f"vidx_base_{vi}") for vi in range(n_vectors)]
        v_val_base = [self.alloc(f"vval_base_{vi}") for vi in range(n_vectors)]
        
        # Double buffer for 6 vectors (node values and addresses)
        v_node_A = [self.alloc(f"v_node_A{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_node_B = [self.alloc(f"v_node_B{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_addr_A = [self.alloc(f"v_addr_A{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_addr_B = [self.alloc(f"v_addr_B{i}", VLEN) for i in range(NUM_PARALLEL)]
        
        # Temp vectors
        v_tmp1 = [self.alloc(f"v_tmp1_{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_tmp2 = [self.alloc(f"v_tmp2_{i}", VLEN) for i in range(NUM_PARALLEL)]
        
        # Preloaded nodes
        v_node_r0 = self.alloc("v_node_r0", VLEN)
        v_node_r1_diff = self.alloc("v_node_r1_diff", VLEN)
        v_node_r1_2 = self.alloc("v_node_r1_2", VLEN)
        v_node_r2 = [self.alloc(f"v_node_r2_{i}", VLEN) for i in range(4)]
        v_r2_diff01 = self.alloc("v_r2_diff01", VLEN)
        v_r2_diff23 = self.alloc("v_r2_diff23", VLEN)
        
        print(f"Scratch used: {self.scratch_ptr}")
        
        # === INITIALIZATION ===
        self.instrs.append({"load": [("const", tmp[0], 1), ("const", tmp[1], 4)]})
        self.instrs.append({"load": [("load", n_nodes_s, tmp[0]), ("load", forest_p, tmp[1])]})
        self.instrs.append({"load": [("const", tmp[0], 5), ("const", tmp[1], 6)]})
        self.instrs.append({"load": [("load", idx_p, tmp[0]), ("load", val_p, tmp[1])]})
        self.instrs.append({"load": [("const", one_s, 1), ("const", two_s, 2)]})
        
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            aux_addr, aux_val = hash_v_aux[hi]
            self.instrs.append({"load": [("const", tmp[0], val1), ("const", tmp[1], aux_val)]})
            self.instrs.append({"valu": [("vbroadcast", hash_v_c[hi], tmp[0]), ("vbroadcast", aux_addr, tmp[1])]})
        
        self.instrs.append({"valu": [
            ("vbroadcast", v_one, one_s),
            ("vbroadcast", v_two, two_s),
            ("vbroadcast", v_n_nodes, n_nodes_s)
        ]})
        self.instrs.append({"load": [("const", tmp[0], 3)]})
        self.instrs.append({"valu": [("vbroadcast", v_three, tmp[0])]})
        
        # Preload tree nodes
        self.instrs.append({"load": [("load", tmp[0], forest_p)]})
        self.instrs.append({"alu": [("+", tmp[2], forest_p, one_s), ("+", tmp[3], forest_p, two_s)]})
        self.instrs.append({"load": [("load", tmp[1], tmp[2]), ("load", tmp[4], tmp[3])]})
        self.instrs.append({"valu": [("vbroadcast", v_node_r0, tmp[0]), ("vbroadcast", v_node_r1_2, tmp[4])]})
        self.instrs.append({"valu": [("vbroadcast", v_tmp1[0], tmp[1])]})
        self.instrs.append({"valu": [("-", v_node_r1_diff, v_tmp1[0], v_node_r1_2)]})
        
        self.instrs.append({"load": [("const", tmp[5], 3), ("const", tmp[6], 4)]})
        self.instrs.append({"load": [("const", tmp[7], 5), ("const", tmp[8], 6)]})
        self.instrs.append({"alu": [
            ("+", tmp[5], forest_p, tmp[5]), ("+", tmp[6], forest_p, tmp[6]),
            ("+", tmp[7], forest_p, tmp[7]), ("+", tmp[8], forest_p, tmp[8])
        ]})
        self.instrs.append({"load": [("load", tmp[9], tmp[5]), ("load", tmp[10], tmp[6])]})
        self.instrs.append({"load": [("load", tmp[11], tmp[7]), ("load", tmp[12], tmp[8])]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_node_r2[0], tmp[9]),
            ("vbroadcast", v_node_r2[1], tmp[10]),
            ("vbroadcast", v_node_r2[2], tmp[11]),
            ("vbroadcast", v_node_r2[3], tmp[12])
        ]})
        self.instrs.append({"valu": [
            ("-", v_r2_diff01, v_node_r2[1], v_node_r2[0]),
            ("-", v_r2_diff23, v_node_r2[3], v_node_r2[2])
        ]})
        
        for vi in range(n_vectors):
            self.instrs.append({"load": [("const", tmp[0], vi * VLEN)]})
            self.instrs.append({"alu": [("+", v_idx_base[vi], idx_p, tmp[0]), ("+", v_val_base[vi], val_p, tmp[0])]})
        
        for vi in range(n_vectors):
            self.instrs.append({"load": [("vload", indices[vi], v_idx_base[vi]), ("vload", values[vi], v_val_base[vi])]})
        
        self.instrs.append({"flow": [("pause",)]})
        
        # === HASH HELPER ===
        def build_hash_ops(vi_list, v_node):
            """Build valu ops for hashing. Returns list of valu op lists."""
            n = len(vi_list)
            ops = []
            
            ops.append([("^", values[vi_list[p]], values[vi_list[p]], v_node[p]) for p in range(n)])
            
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                c_v = hash_v_c[hi]
                aux_v = hash_v_aux[hi][0]
                
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    ops.append([("multiply_add", values[vi_list[p]], values[vi_list[p]], aux_v, c_v) for p in range(n)])
                else:
                    ops1 = []
                    for p in range(n):
                        ops1.append((op1, v_tmp1[p], values[vi_list[p]], c_v))
                        ops1.append((op3, v_tmp2[p], values[vi_list[p]], aux_v))
                    for i in range(0, len(ops1), 6):
                        ops.append(ops1[i:i+6])
                    ops.append([(op2, values[vi_list[p]], v_tmp1[p], v_tmp2[p]) for p in range(n)])
            
            ops.append([("&", v_tmp1[p], values[vi_list[p]], v_one) for p in range(n)])
            ops.append([("+", v_tmp1[p], v_tmp1[p], v_one) for p in range(n)])
            ops.append([("multiply_add", indices[vi_list[p]], indices[vi_list[p]], v_two, v_tmp1[p]) for p in range(n)])
            ops.append([("<", v_tmp1[p], indices[vi_list[p]], v_n_nodes) for p in range(n)])
            ops.append([("*", indices[vi_list[p]], indices[vi_list[p]], v_tmp1[p]) for p in range(n)])
            
            return ops
        
        def emit_hash(vi_list, v_node):
            """Emit hash without interleaving."""
            for op_list in build_hash_ops(vi_list, v_node):
                self.instrs.append({"valu": op_list})
        
        def emit_addr_calc(vi_list, v_addr):
            n = len(vi_list)
            self.instrs.append({"valu": [("vbroadcast", v_addr[p], forest_p) for p in range(n)]})
            self.instrs.append({"valu": [("+", v_addr[p], v_addr[p], indices[vi_list[p]]) for p in range(n)]})
        
        def emit_gather_with_hash(gather_vi, v_node_g, v_addr_g, hash_vi, v_node_h):
            """Interleaved gather + hash."""
            n_g = len(gather_vi)
            n_h = len(hash_vi) if hash_vi else 0
            
            load_ops = []
            for p in range(n_g):
                for lane in range(VLEN):
                    load_ops.append(("load", v_node_g[p] + lane, v_addr_g[p] + lane))
            
            hash_ops = build_hash_ops(hash_vi, v_node_h) if n_h > 0 else []
            
            load_idx = 0
            hash_idx = 0
            
            while load_idx < len(load_ops) or hash_idx < len(hash_ops):
                instr = {}
                if load_idx < len(load_ops):
                    loads = []
                    for _ in range(2):
                        if load_idx < len(load_ops):
                            loads.append(load_ops[load_idx])
                            load_idx += 1
                    instr["load"] = loads
                if hash_idx < len(hash_ops):
                    instr["valu"] = hash_ops[hash_idx]
                    hash_idx += 1
                if instr:
                    self.instrs.append(instr)
        
        # === MAIN LOOP ===
        for r in range(rounds):
            if r == 0:
                for gi in range(n_groups):
                    vi_list = get_vi_list(gi)
                    v_node = [v_node_r0] * len(vi_list)
                    emit_hash(vi_list, v_node)
                    
            elif r == 1:
                for gi in range(n_groups):
                    vi_list = get_vi_list(gi)
                    n = len(vi_list)
                    # Compute node values into v_node_A (safe buffer)
                    self.instrs.append({"valu": [("&", v_node_A[p], indices[vi_list[p]], v_one) for p in range(n)]})
                    self.instrs.append({"valu": [("multiply_add", v_node_A[p], v_node_A[p], v_node_r1_diff, v_node_r1_2) for p in range(n)]})
                    emit_hash(vi_list, v_node_A[:n])
                    
            elif r == 2:
                for gi in range(n_groups):
                    vi_list = get_vi_list(gi)
                    n = len(vi_list)
                    
                    # rel = idx - 3 -> v_tmp2
                    self.instrs.append({"valu": [("-", v_tmp2[p], indices[vi_list[p]], v_three) for p in range(n)]})
                    # bit0 = rel & 1 -> v_node_B
                    self.instrs.append({"valu": [("&", v_node_B[p], v_tmp2[p], v_one) for p in range(n)]})
                    # bit1 = 1 - (rel < 2)
                    self.instrs.append({"valu": [("<", v_tmp2[p], v_tmp2[p], v_two) for p in range(n)]})
                    self.instrs.append({"valu": [("-", v_tmp2[p], v_one, v_tmp2[p]) for p in range(n)]})
                    # pair0 = bit0 * diff01 + node3 -> v_node_A
                    self.instrs.append({"valu": [("multiply_add", v_node_A[p], v_node_B[p], v_r2_diff01, v_node_r2[0]) for p in range(n)]})
                    # pair1 = bit0 * diff23 + node5 -> v_node_B
                    self.instrs.append({"valu": [("multiply_add", v_node_B[p], v_node_B[p], v_r2_diff23, v_node_r2[2]) for p in range(n)]})
                    # diff = pair1 - pair0
                    self.instrs.append({"valu": [("-", v_node_B[p], v_node_B[p], v_node_A[p]) for p in range(n)]})
                    # result = bit1 * diff + pair0 -> v_node_A
                    self.instrs.append({"valu": [("multiply_add", v_node_A[p], v_tmp2[p], v_node_B[p], v_node_A[p]) for p in range(n)]})
                    emit_hash(vi_list, v_node_A[:n])
                    
            else:
                # Rounds 3+: Software pipelined gather
                pending_vi = None
                pending_v_node = None
                
                for gi in range(n_groups):
                    vi_list = get_vi_list(gi)
                    
                    if gi % 2 == 0:
                        v_node = v_node_A
                        v_addr = v_addr_A
                    else:
                        v_node = v_node_B
                        v_addr = v_addr_B
                    
                    emit_addr_calc(vi_list, v_addr)
                    emit_gather_with_hash(vi_list, v_node, v_addr,
                                          pending_vi if pending_vi else [],
                                          pending_v_node)
                    
                    pending_vi = vi_list
                    pending_v_node = v_node[:len(vi_list)]
                
                if pending_vi:
                    emit_hash(pending_vi, pending_v_node)
        
        # === STORE ===
        for vi in range(n_vectors):
            self.instrs.append({"store": [("vstore", v_idx_base[vi], indices[vi]), ("vstore", v_val_base[vi], values[vi])]})
        
        self.instrs.append({"flow": [("pause",)]})


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
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
