from heapq import heappop, heappush
import numpy as np
from search import IPSearch, IPNode2, Utilities as U
from realkd.search import BestBoundFirstBoundary
import copy
from bitarray import bitarray
from math import inf
from sortedcontainers import SortedSet
import sortednp as snp


class BestBoundFirstBoundary:

    def __init__(self):
        self.heap = []

    def __bool__(self):
        return bool(self.heap)

    def __len__(self):
        return len(self.heap)

    def push(self, augmented_node):
        _, node = augmented_node
        heappush(self.heap, (-node.val_bound, -node.val, augmented_node))

    def pop(self):
        _, _, augmented_node = heappop(self.heap)
        return augmented_node


class BFS(IPSearch): # best first search
    def __init__(self, root, curr_node, heap, context, res, obj, bnd):
        super().__init__(root, curr_node, heap, context, res, obj, bnd)

    def push_children(self, curr_node, j):
        # j = curr_node.active_attr
        if not curr_node.intent.fully_closed(j):
            new_locked_attrs = np.copy(curr_node.locked_attrs)
            new_locked_attrs[j] = 1
            minus_upper = curr_node.get_minus_upper(j)

            self.res.num_candidates += 1
            minus_upper.obj_val = self.obj(minus_upper.extension.indices)
            minus_upper.bnd_val = self.bnd(minus_upper.extension.indices)
            heappush(self.heap, minus_upper)

            if not curr_node.locked_attrs[j]: # if j is not a locked attribute
                plus_lower = curr_node.get_plus_lower(j)
                #potentially add a check here
                self.res.num_candidates += 1
                plus_lower.obj_val = self.obj(plus_lower.extension.indices)
                plus_lower.bnd_val = self.bnd(plus_lower.extension.indices)
                heappush(self.heap, plus_lower)

    # def push_shift(self, curr_node):
    #     shifted_node = copy.deepcopy(curr_node)
    #     shifted_node.active_attr -= 1
    #     shifted_node.shifted = True
    #     heappush(self.heap, shifted_node)

    def traversal(self):
        boundary = BestBoundFirstBoundary()
        full = self.ctx.extension([])
        full_bits = bitarray(len(full))
        full_bits.setall(1) # is this necessary?
        max_bnd = self.bnd(full)
        root_node = IPNode2(SortedSet([]), bitarray((0 for _ in range(self.ctx.n))), full, full_bits, -1, self.ctx.n, self.obj(full), self.bnd(full))
        opt_node = root_node
        yield root_node

        boundary.push(root_node)
        
        while boundary: # while queue is not empty
            ops, curr_node = boundary.pop()
            obj_val = self.obj(extension)
            bnd_val = self.bnd(extension)
            
            j = curr_node.active_attr
            # self.res.max_obj = max(self.curr_node.obj_val, self.res.max_obj)
            if obj_val >= opt_node.val:
                    self.res.max_obj = curr_node.obj_val
                    opt_node = curr_node
            if curr_node.bnd_val > self.res.max_obj: # or self.curr_node == self.root: # root node check because bnd > max_obj check fails on root
                closed_intent = curr_node.extension.get_closure() # maybe make this conditional to avoid repeat closure
                if U.is_canonical(curr_node.intent, closed_intent, j):
                    curr_node.intent = closed_intent
                    self.res.num_nodes += 1
                    if self.res.max_obj == max_bnd:
                        break
                    self.push_children(curr_node, j-1)