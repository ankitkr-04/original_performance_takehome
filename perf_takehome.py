"""
# Anthropic's Original Performance Engineering Take-home (Release version)
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
        TRUE software-pipelined kernel with OVERLAPPED load and compute.
        
        Key insight: In steady state, each cycle does BOTH:
        - 2 scalar loads (gathering node values for group G+1)
        - Up to 6 valu ops (hashing group G)
        
        We process 6 vectors at a time (to use all 6 valu slots).
        Each vector has 8 lanes = 8 loads needed.
        With 2 loads/cycle, 6 vectors need 6*8/2 = 24 cycles to gather.
        Hash computation per 6 vectors = ~18 valu cycles.
        So gathering dominates - we fill valu gaps with hash ops.
        """
        n_vectors = batch_size // VLEN  # 32 vectors
        NUM_PARALLEL = 6  # 6 vectors to use all 6 valu slots
        n_groups = n_vectors // NUM_PARALLEL  # 5 groups (with 2 leftover vectors)
        leftover = n_vectors % NUM_PARALLEL  # 2 leftover vectors
        
        # === SCRATCH ALLOCATION ===
        tmp = [self.alloc(f"tmp{i}") for i in range(12)]  # temp scalars
        
        forest_p = self.alloc("forest_p")
        idx_p = self.alloc("idx_p")
        val_p = self.alloc("val_p")
        n_nodes_s = self.alloc("n_nodes_s")
        
        one_s = self.alloc("one_s")
        two_s = self.alloc("two_s")
        
        # Precomputed vector base addresses for each vector
        v_idx_base = [self.alloc(f"vidx_base_{vi}") for vi in range(n_vectors)]
        v_val_base = [self.alloc(f"vval_base_{vi}") for vi in range(n_vectors)]
        
        v_one = self.alloc("v_one", VLEN)
        v_two = self.alloc("v_two", VLEN)
        v_n_nodes = self.alloc("v_n_nodes", VLEN)
        
        # Hash constants - pre-broadcast to vectors
        hash_v_c = []
        hash_v_aux = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_v_c.append(self.alloc(f"hvc_{hi}", VLEN))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                aux_val = 1 + (1 << val3)
            else:
                aux_val = val3
            hash_v_aux.append((self.alloc(f"hva_{hi}", VLEN), aux_val))
        
        indices = [self.alloc(f"idx_{i}", VLEN) for i in range(n_vectors)]
        values = [self.alloc(f"val_{i}", VLEN) for i in range(n_vectors)]
        
        # Triple buffer for pipelining: A, B, C
        v_node_A = [self.alloc(f"v_node_A{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_node_B = [self.alloc(f"v_node_B{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_addr_A = [self.alloc(f"v_addr_A{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_addr_B = [self.alloc(f"v_addr_B{i}", VLEN) for i in range(NUM_PARALLEL)]
        
        # Temp vectors for hash computation
        v_tmp1 = [self.alloc(f"v_tmp1_{i}", VLEN) for i in range(NUM_PARALLEL)]
        v_tmp2 = [self.alloc(f"v_tmp2_{i}", VLEN) for i in range(NUM_PARALLEL)]
        
        # Preloaded tree nodes for early rounds (to avoid gather)
        # Round 0: all indices=0, so just need node[0] broadcast
        v_node_r0 = self.alloc("v_node_r0", VLEN)  # broadcast of node[0]
        # Round 1: indices are 1 or 2, preload both
        v_node_r1_1 = self.alloc("v_node_r1_1", VLEN)  # node[1]
        v_node_r1_2 = self.alloc("v_node_r1_2", VLEN)  # node[2]
        v_node_r1_diff = self.alloc("v_node_r1_diff", VLEN)  # node[1] - node[2]
        # Round 2: indices are 3,4,5,6, preload all 4
        v_node_r2 = [self.alloc(f"v_node_r2_{i}", VLEN) for i in range(4)]  # nodes 3,4,5,6
        
        # === INITIALIZATION ===
        # Load header values
        self.instrs.append({"load": [("const", tmp[0], 1), ("const", tmp[1], 4)]})
        self.instrs.append({"load": [("load", n_nodes_s, tmp[0]), ("load", forest_p, tmp[1])]})
        self.instrs.append({"load": [("const", tmp[0], 5), ("const", tmp[1], 6)]})
        self.instrs.append({"load": [("load", idx_p, tmp[0]), ("load", val_p, tmp[1])]})
        
        self.instrs.append({"load": [("const", one_s, 1), ("const", two_s, 2)]})
        
        # Load hash constants using scalar temps then broadcast
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            aux_addr, aux_val = hash_v_aux[hi]
            self.instrs.append({"load": [("const", tmp[0], val1), ("const", tmp[1], aux_val)]})
            self.instrs.append({"valu": [
                ("vbroadcast", hash_v_c[hi], tmp[0]),
                ("vbroadcast", aux_addr, tmp[1])
            ]})
        
        # Broadcast scalar constants to vectors
        self.instrs.append({"valu": [
            ("vbroadcast", v_one, one_s),
            ("vbroadcast", v_two, two_s),
            ("vbroadcast", v_n_nodes, n_nodes_s)
        ]})
        
        # Preload tree nodes for rounds 0-2
        # Round 0: broadcast node[0]
        self.instrs.append({"load": [("load", tmp[0], forest_p)]})  # tmp[0] = node[0]
        self.instrs.append({"valu": [("vbroadcast", v_node_r0, tmp[0])]})
        
        # Round 1: load nodes 1 and 2, broadcast, compute diff
        self.instrs.append({"alu": [("+", tmp[2], forest_p, one_s), ("+", tmp[3], forest_p, two_s)]})
        self.instrs.append({"load": [("load", tmp[0], tmp[2]), ("load", tmp[1], tmp[3])]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_node_r1_1, tmp[0]),
            ("vbroadcast", v_node_r1_2, tmp[1])
        ]})
        self.instrs.append({"valu": [("-", v_node_r1_diff, v_node_r1_1, v_node_r1_2)]})
        
        # Round 2: load nodes 3,4,5,6
        self.instrs.append({"load": [("const", tmp[4], 3), ("const", tmp[5], 4)]})
        self.instrs.append({"load": [("const", tmp[6], 5), ("const", tmp[7], 6)]})
        self.instrs.append({"alu": [
            ("+", tmp[4], forest_p, tmp[4]), ("+", tmp[5], forest_p, tmp[5]),
            ("+", tmp[6], forest_p, tmp[6]), ("+", tmp[7], forest_p, tmp[7])
        ]})
        self.instrs.append({"load": [("load", tmp[0], tmp[4]), ("load", tmp[1], tmp[5])]})
        self.instrs.append({"load": [("load", tmp[2], tmp[6]), ("load", tmp[3], tmp[7])]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_node_r2[0], tmp[0]),
            ("vbroadcast", v_node_r2[1], tmp[1]),
            ("vbroadcast", v_node_r2[2], tmp[2]),
            ("vbroadcast", v_node_r2[3], tmp[3])
        ]})
        
        # Compute base addresses for each vector's idx and val arrays
        for vi in range(n_vectors):
            self.instrs.append({"load": [("const", tmp[0], vi * VLEN)]})
            self.instrs.append({"alu": [
                ("+", v_idx_base[vi], idx_p, tmp[0]),
                ("+", v_val_base[vi], val_p, tmp[0])
            ]})
        
        # Load initial indices and values
        for vi in range(n_vectors):
            self.instrs.append({"load": [
                ("vload", indices[vi], v_idx_base[vi]),
                ("vload", values[vi], v_val_base[vi])
            ]})
        
        self.instrs.append({"flow": [("pause",)]})
        
        # === MAIN LOOP with TRUE software pipelining ===
        
        def get_vi_list(gi):
            """Get vector indices for group gi"""
            vi_base = gi * NUM_PARALLEL
            return list(range(vi_base, min(vi_base + NUM_PARALLEL, n_vectors)))
        
        def emit_addr_calc(vi_list, v_addr):
            """Emit address calculation: v_addr[p] = forest_p + indices[vi_list[p]]"""
            n = len(vi_list)
            # Broadcast forest_p and add indices
            self.instrs.append({"valu": [
                ("vbroadcast", v_addr[p], forest_p) for p in range(min(n, 6))
            ]})
            self.instrs.append({"valu": [
                ("+", v_addr[p], v_addr[p], indices[vi_list[p]]) for p in range(min(n, 6))
            ]})
        
        def build_hash_broadcast(vi_list, v_node_bcast):
            """Build hash ops for round 0 where all items use same broadcast node value."""
            n = len(vi_list)
            ops = []
            
            # XOR with broadcast node value (same for all)
            ops.append([("^", values[vi_list[p]], values[vi_list[p]], v_node_bcast) for p in range(n)])
            
            # Hash stages (same as regular)
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
            
            # Index update
            ops.append([("&", v_tmp1[p], values[vi_list[p]], v_one) for p in range(n)])
            ops.append([("+", v_tmp1[p], v_tmp1[p], v_one) for p in range(n)])
            ops.append([("multiply_add", indices[vi_list[p]], indices[vi_list[p]], v_two, v_tmp1[p]) for p in range(n)])
            
            # Wrap check
            ops.append([("<", v_tmp1[p], indices[vi_list[p]], v_n_nodes) for p in range(n)])
            ops.append([("*", indices[vi_list[p]], indices[vi_list[p]], v_tmp1[p]) for p in range(n)])
            
            return ops
        
        def build_hash_round1(vi_list):
            """Build hash ops for round 1 with arithmetic mux: node = diff*(idx&1) + node2."""
            n = len(vi_list)
            ops = []
            
            # Compute node value using arithmetic: node = diff*(idx&1) + node2
            # idx is 1 or 2; idx&1 is 1 or 0
            # If idx=1: node = diff*1 + node2 = node1
            # If idx=2: node = diff*0 + node2 = node2
            ops.append([("&", v_tmp1[p], indices[vi_list[p]], v_one) for p in range(n)])
            ops.append([("multiply_add", v_tmp1[p], v_node_r1_diff, v_tmp1[p], v_node_r1_2) for p in range(n)])
            
            # XOR with computed node
            ops.append([("^", values[vi_list[p]], values[vi_list[p]], v_tmp1[p]) for p in range(n)])
            
            # Hash stages
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
            
            # Index update
            ops.append([("&", v_tmp1[p], values[vi_list[p]], v_one) for p in range(n)])
            ops.append([("+", v_tmp1[p], v_tmp1[p], v_one) for p in range(n)])
            ops.append([("multiply_add", indices[vi_list[p]], indices[vi_list[p]], v_two, v_tmp1[p]) for p in range(n)])
            
            # Wrap check
            ops.append([("<", v_tmp1[p], indices[vi_list[p]], v_n_nodes) for p in range(n)])
            ops.append([("*", indices[vi_list[p]], indices[vi_list[p]], v_tmp1[p]) for p in range(n)])
            
            return ops
        
        def build_hash_round2(vi_list):
            """Build hash ops for round 2 with 2-bit arithmetic selection from 4 nodes (3,4,5,6)."""
            n = len(vi_list)
            ops = []
            
            # idx is in range 3-6. We need to select from v_node_r2[0..3]
            # idx-3 gives us 0,1,2,3. Use 2-bit selection.
            # bit0 = (idx-3) & 1, bit1 = ((idx-3) >> 1) & 1
            # 
            # Level 0 pairs: select between (n3,n4) and (n5,n6) based on bit0
            #   pair0 = bit0*(n4-n3) + n3  (selects n3 if bit0=0, n4 if bit0=1) -- WRONG
            # Actually: nodes are at indices 3,4,5,6
            #   idx=3 -> idx-3=0 -> bit1=0,bit0=0 -> want node[3] = v_node_r2[0]
            #   idx=4 -> idx-3=1 -> bit1=0,bit0=1 -> want node[4] = v_node_r2[1]
            #   idx=5 -> idx-3=2 -> bit1=1,bit0=0 -> want node[5] = v_node_r2[2]
            #   idx=6 -> idx-3=3 -> bit1=1,bit0=1 -> want node[6] = v_node_r2[3]
            # 
            # Using multiply_add: result = a*b + c
            # For bit0 selection between n0 and n1: result = bit0*(n1-n0) + n0
            #   bit0=0: result = n0
            #   bit0=1: result = n1-n0+n0 = n1
            
            # First compute idx-3 (use tmp2 since we need tmp1 for bit0)
            # We need a constant 3 vector - we can compute it as v_two + v_one
            ops.append([("+", v_tmp2[p], v_two, v_one) for p in range(n)])  # v_tmp2 = 3
            ops.append([("-", v_tmp2[p], indices[vi_list[p]], v_tmp2[p]) for p in range(n)])  # v_tmp2 = idx-3
            
            # bit0 = (idx-3) & 1
            ops.append([("&", v_tmp1[p], v_tmp2[p], v_one) for p in range(n)])  # v_tmp1 = bit0
            
            # bit1 = ((idx-3) >> 1) & 1 = (idx-3) / 2 (integer division)
            # But we don't have shift. We can use: bit1 = (idx-3 - bit0) / 2
            # Actually simpler: bit1 = (idx >= 5) = (idx-3 >= 2)
            # Or: first do level 0 selection, store in tmp, then do level 1
            
            # Level 0: select pairs based on bit0
            # pair0 = bit0*(n1-n0) + n0 where n0=v_node_r2[0], n1=v_node_r2[1]
            # pair1 = bit0*(n3-n2) + n2 where n2=v_node_r2[2], n3=v_node_r2[3]
            # 
            # New plan:
            # - v_node_B[0] = diff01, v_node_B[1] = diff23 (temps)
            # - v_node_A[p] = pair0[p]
            # - v_tmp1 will hold bit0, then reuse for pair1[p]
            # - v_tmp2 will hold idx-3, then bit1
            
            # Compute diffs in v_node_B
            ops.append([("-", v_node_B[0], v_node_r2[1], v_node_r2[0]), ("-", v_node_B[1], v_node_r2[3], v_node_r2[2])])
            
            # Compute bit1 before overwriting v_tmp2
            ops.append([("<", v_tmp2[p], v_tmp2[p], v_two) for p in range(n)])  # v_tmp2 = (idx-3 < 2)
            ops.append([("-", v_tmp2[p], v_one, v_tmp2[p]) for p in range(n)])  # v_tmp2 = bit1
            
            # pair0[p] = bit0[p] * diff01 + n0 (store in v_node_A)
            ops.append([("multiply_add", v_node_A[p], v_tmp1[p], v_node_B[0], v_node_r2[0]) for p in range(n)])  # pair0
            
            # pair1[p] = bit0[p] * diff23 + n2 (reuse v_tmp1 for pair1)
            ops.append([("multiply_add", v_tmp1[p], v_tmp1[p], v_node_B[1], v_node_r2[2]) for p in range(n)])  # pair1 in v_tmp1
            
            # diff_pairs = pair1 - pair0 (store in v_node_B[p])
            ops.append([("-", v_node_B[p], v_tmp1[p], v_node_A[p]) for p in range(n)])  # diff_pairs in v_node_B
            
            # result = bit1 * diff_pairs + pair0 (store in v_tmp1)
            ops.append([("multiply_add", v_tmp1[p], v_tmp2[p], v_node_B[p], v_node_A[p]) for p in range(n)])  # final node in v_tmp1
            
            # XOR with computed node
            ops.append([("^", values[vi_list[p]], values[vi_list[p]], v_tmp1[p]) for p in range(n)])
            
            # Hash stages
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
            
            # Index update
            ops.append([("&", v_tmp1[p], values[vi_list[p]], v_one) for p in range(n)])
            ops.append([("+", v_tmp1[p], v_tmp1[p], v_one) for p in range(n)])
            ops.append([("multiply_add", indices[vi_list[p]], indices[vi_list[p]], v_two, v_tmp1[p]) for p in range(n)])
            
            # Wrap check
            ops.append([("<", v_tmp1[p], indices[vi_list[p]], v_n_nodes) for p in range(n)])
            ops.append([("*", indices[vi_list[p]], indices[vi_list[p]], v_tmp1[p]) for p in range(n)])
            
            return ops

        def build_hash_valu_ops(vi_list, v_node):
            """Build list of all valu ops needed for one group's hash.
            Returns list of (list of valu ops) - each inner list ≤ 6 ops."""
            n = len(vi_list)
            ops = []
            
            # XOR: value ^= node_val
            ops.append([("^", values[vi_list[p]], values[vi_list[p]], v_node[p]) for p in range(n)])
            
            # Hash stages
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                c_v = hash_v_c[hi]
                aux_v = hash_v_aux[hi][0]
                
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    # Optimized: value = value * (1 + (1 << val3)) + c
                    ops.append([("multiply_add", values[vi_list[p]], values[vi_list[p]], aux_v, c_v) for p in range(n)])
                else:
                    # General: tmp1 = op1(value, c), tmp2 = op3(value, aux), value = op2(tmp1, tmp2)
                    # Split into ≤6 ops per cycle
                    ops1 = []
                    for p in range(n):
                        ops1.append((op1, v_tmp1[p], values[vi_list[p]], c_v))
                        ops1.append((op3, v_tmp2[p], values[vi_list[p]], aux_v))
                    # Emit in chunks of 6
                    for i in range(0, len(ops1), 6):
                        ops.append(ops1[i:i+6])
                    
                    # Combine
                    ops.append([(op2, values[vi_list[p]], v_tmp1[p], v_tmp2[p]) for p in range(n)])
            
            # Index update: idx = 2*idx + (1 if (val&1)==0 else 2) = 2*idx + 1 + (val&1)
            # But actually: branch = (val&1)+1, idx = idx*2 + branch
            ops.append([("&", v_tmp1[p], values[vi_list[p]], v_one) for p in range(n)])
            ops.append([("+", v_tmp1[p], v_tmp1[p], v_one) for p in range(n)])
            ops.append([("multiply_add", indices[vi_list[p]], indices[vi_list[p]], v_two, v_tmp1[p]) for p in range(n)])
            
            # Wrap check: if idx >= n_nodes, idx = 0
            ops.append([("<", v_tmp1[p], indices[vi_list[p]], v_n_nodes) for p in range(n)])
            ops.append([("*", indices[vi_list[p]], indices[vi_list[p]], v_tmp1[p]) for p in range(n)])
            
            return ops
        
        def emit_gather_with_hash(gather_vi_list, v_node_gather, v_addr_gather, hash_vi_list, v_node_hash):
            """
            Emit interleaved gather and hash operations.
            Gather: Load node values for gather_vi_list using v_addr_gather into v_node_gather
            Hash: Compute hash for hash_vi_list using v_node_hash
            
            We have 2 load slots and 6 valu slots per cycle.
            Gather needs 8 loads per vector * n_vectors / 2 = cycles
            Hash needs ~18 valu cycles for 6 vectors.
            """
            n_gather = len(gather_vi_list)
            
            # Build load sequence: for each vector, load all 8 lanes
            load_ops = []
            for p in range(n_gather):
                for lane in range(VLEN):
                    load_ops.append(("load", v_node_gather[p] + lane, v_addr_gather[p] + lane))
            
            # Build hash valu ops
            if hash_vi_list:
                hash_ops = build_hash_valu_ops(hash_vi_list, v_node_hash)
            else:
                hash_ops = []
            
            # Interleave: 2 loads per cycle, up to 6 valus per cycle
            load_idx = 0
            hash_idx = 0
            
            while load_idx < len(load_ops) or hash_idx < len(hash_ops):
                instr = {}
                
                # Add up to 2 loads
                if load_idx < len(load_ops):
                    loads = []
                    for _ in range(2):
                        if load_idx < len(load_ops):
                            loads.append(load_ops[load_idx])
                            load_idx += 1
                    instr["load"] = loads
                
                # Add up to 6 valus (one hash step)
                if hash_idx < len(hash_ops):
                    instr["valu"] = hash_ops[hash_idx]
                    hash_idx += 1
                
                self.instrs.append(instr)
        
        total_groups = n_groups + (1 if leftover > 0 else 0)
        
        # Track pending hash from previous gather
        pending_hash_vi_list = None
        pending_hash_v_node = None
        pending_hash_round = None
        
        def emit_pending_hash():
            """Emit any pending hash operations (no overlap with gather)."""
            nonlocal pending_hash_vi_list, pending_hash_v_node, pending_hash_round
            if pending_hash_vi_list:
                if pending_hash_round == 0:
                    hash_ops = build_hash_broadcast(pending_hash_vi_list, v_node_r0)
                elif pending_hash_round == 1:
                    hash_ops = build_hash_round1(pending_hash_vi_list)
                elif pending_hash_round == 2:
                    hash_ops = build_hash_round2(pending_hash_vi_list)
                else:
                    hash_ops = build_hash_valu_ops(pending_hash_vi_list, pending_hash_v_node)
                for ops in hash_ops:
                    self.instrs.append({"valu": ops})
                pending_hash_vi_list = None
                pending_hash_v_node = None
                pending_hash_round = None
        
        for r in range(rounds):
            if r <= 2:
                # Rounds 0, 1, 2: Use preloaded nodes, no gather needed
                # Just emit hash for all groups sequentially (no gather to overlap with)
                # First, flush any pending hash from previous round
                emit_pending_hash()
                
                for gi in range(total_groups):
                    vi_list = get_vi_list(gi)
                    if r == 0:
                        hash_ops = build_hash_broadcast(vi_list, v_node_r0)
                    elif r == 1:
                        hash_ops = build_hash_round1(vi_list)
                    else:  # r == 2
                        hash_ops = build_hash_round2(vi_list)
                    for ops in hash_ops:
                        self.instrs.append({"valu": ops})
            else:
                # Rounds 3+: Need to gather node values
                for gi in range(total_groups):
                    vi_list = get_vi_list(gi)
                    
                    # Use alternating buffers
                    if gi % 2 == 0:
                        v_node = v_node_A
                        v_addr = v_addr_A
                    else:
                        v_node = v_node_B
                        v_addr = v_addr_B
                    
                    # Compute addresses for current group
                    emit_addr_calc(vi_list, v_addr)
                    
                    # Gather current group, overlap with pending hash (if any)
                    emit_gather_with_hash(vi_list, v_node, v_addr, 
                                          pending_hash_vi_list if pending_hash_vi_list else [], 
                                          pending_hash_v_node)
                    
                    # Current group becomes pending for next iteration
                    pending_hash_vi_list = vi_list
                    pending_hash_v_node = v_node
                    pending_hash_round = r
        
        # Hash the final pending group (no more gathers to overlap)
        emit_pending_hash()
        
        # Store results BEFORE pause
        for vi in range(n_vectors):
            self.instrs.append({"store": [
                ("vstore", v_idx_base[vi], indices[vi]),
                ("vstore", v_val_base[vi], values[vi])
            ]})
        
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
