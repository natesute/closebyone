import pandas as pd
from heapq import heappop, heappush
from sortedcontainers import SortedSet
from collections import defaultdict
import numpy as np
from math import inf
import sortednp as snp
from realkd.search import BestBoundFirstBoundary, Node
from bitarray import bitarray
from bitarray.util import subset

class BinTreeSearch:
    
    def __init__(self, ctx, obj, bnd, res, order='bestboundfirst'):

        self.ctx = ctx
        self.f = obj
        self.g = bnd
        self.res = res
        self.order = order
        
        # switches
        self.crit_propagation = True

    def traversal(self):
        boundary = BestBoundFirstBoundary()
        full = self.ctx.extension([])
        full_bits = bitarray(len(full))
        full_bits.setall(1)
        root_node = Node(SortedSet([]), bitarray((0 for _ in range(self.ctx.n))), full, full_bits, -1, self.ctx.n, self.f(full), inf)
        opt_node = root_node
        yield root_node

        boundary.push(([(i, self.ctx.n, inf) for i in range(self.ctx.n)], root_node))

        while boundary:
            ops, current = boundary.pop()
            self.res.num_nodes += 1
            children = []
            # for a in ops:
            for aug, crit, bnd in ops:
                self.res.num_candidates += 1 # is this the right place to have this?
                if aug <= current.gen_index:  # need to also check == case it seems
                    continue
                if self.crit_propagation and crit < current.gen_index:
                    continue
                if bnd <= opt_node.val:  # checking old bound against potentially updated opt value
                    continue
                if current.closure[aug]:
                    continue

                extension = snp.intersect(np.array(current.extension, dtype=np.int64), self.ctx.extents[aug])
                val = self.f(extension)
                bound = self.g(extension)

                generator = current.generator[:]
                generator.append(aug)


                if bound < opt_node.val and val <= opt_node.val:
                    continue

                bit_extension = current.bit_extension & self.ctx.bit_extents[aug]
                closure = bitarray(current.closure)
                closure[aug] = True
                if self.crit_propagation and crit < aug and not current.closure[crit]:
                    # aug still needed for descendants but for current is guaranteed
                    # to lead to not lexmin child; hence can recycle current crit index
                    # (as upper bound to real crit index)
                    crit_idx = crit
                else:
                    crit_idx = self.ctx.find_small_crit_index(aug, bit_extension, closure)

                if crit_idx > aug:  # in this case crit_idx == n (sentinel)
                    crit_idx = self.ctx.complete_closure(aug, bit_extension, closure)
                else:
                    closure[crit_idx] = True

                child = Node(generator, closure, extension, bit_extension, aug, crit_idx, val, bound)
                opt_node = max(opt_node, child, key=Node.value)
                yield child

                # early termination if opt value approximately exceeds best active upper bound
                if opt_node.val >= current.val_bound:
                    return

                children += [child]

            augs = []
            for child in children:
                if child.val_bound > opt_node.val:
                    augs.append((child.gen_index, child.crit_idx, child.val_bound))

            for child in children:
                if child.valid:
                    boundary.push((augs, child))

    def run(self):
        opt_node = None
        opt_value = -inf
        k = 0
        for node in self.traversal():
            k += 1
            if opt_value < node.val:
                opt_node = node
                opt_value = node.val

        if not opt_node.valid:
            self.ctx.complete_closure(opt_node.gen_index, opt_node.bit_extension, opt_node.closure)
        # min_generator = self.ctx.greedy_simplification([i for i in range(len(opt_node.closure)) if opt_node.closure[i]], opt_node.extension)

        self.res.opt_node = opt_node
        self.res.opt_value = opt_node.val
        #return Conjunction(map(lambda i: self.ctx.attributes[i], min_generator))