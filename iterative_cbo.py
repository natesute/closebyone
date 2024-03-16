from heapq import heappop, heappush
import numpy as np
from search import Utilities as U, IPNode
import copy


class Search: # best first search
    def __init__(self, heap, ctx, res, obj, bnd):
        self.heap = heap
        self.ctx = ctx
        self.res = res
        self.obj = obj
        self.bnd = bnd

    def push_children(self, curr_node, j):
        # j = curr_node.active_attr
        if not curr_node.intent.fully_closed(j):
            new_locked_attrs = np.copy(curr_node.locked_attrs)
            new_locked_attrs[j] = 1
            minus_upper = curr_node.get_minus_upper(j)

            minus_upper.obj_val = self.obj(minus_upper.extension)
            minus_upper.bnd_val = self.bnd(minus_upper.extension)
            self.res.num_candidates += 1
            heappush(self.heap, minus_upper)

            if not curr_node.locked_attrs[j]: # if j is not a locked attribute
                plus_lower = curr_node.get_plus_lower(j)
                #potentially add a check here
                
                plus_lower.obj_val = self.obj(plus_lower.extension)
                plus_lower.bnd_val = self.bnd(plus_lower.extension)
                self.res.num_candidates += 1
                heappush(self.heap, plus_lower)

    def run(self):
        root_ext = np.arange(self.ctx.n)
        max_bnd = self.bnd(root_ext)
        root_node = IPNode(self.ctx, self.ctx.closure(root_ext), self.ctx.m-1, np.zeros(self.ctx.m, dtype=int))
        root_node.obj_val = self.obj(root_ext)
        root_node.bnd_val = max_bnd

        heappush(self.heap, root_node)
        
        while self.heap: # while queue is not empty

            curr_node = heappop(self.heap)
            curr_node.obj_val = self.obj(curr_node.extension)
            curr_node.bnd_val = self.bnd(curr_node.extension)
            
            j = curr_node.active_attr
            # self.res.max_obj = max(self.curr_node.obj_val, self.res.max_obj)
            if curr_node.obj_val >= self.res.max_obj:
                    self.res.max_obj = curr_node.obj_val
                    self.res.opt_node = curr_node
            if curr_node.bnd_val > self.res.max_obj: # or self.curr_node == self.root: # root node check because bnd > max_obj check fails on root
                closed_intent = self.ctx.closure(curr_node.extension) # maybe make this conditional to avoid repeat closure
                if U.is_canonical(curr_node.intent, closed_intent, j):
                    curr_node.intent = closed_intent
                    self.res.num_nodes += 1
                    if self.res.max_obj == max_bnd:
                        break
                    self.push_children(curr_node, j)
                if j > 0:
                    self.push_children(curr_node, j-1)

class Search2(IPSearch): # best first search
    def __init__(self, heap, ctx, res, obj, bnd):
        super().__init__( heap, ctx, res, obj, bnd)


    def push_children(self, curr_node):
        j = curr_node.active_attr
        if not curr_node.intent.fully_closed(j):
            new_locked_attrs = np.copy(curr_node.locked_attrs)
            new_locked_attrs[j] = 1
            minus_upper = curr_node.get_minus_upper(j)

            self.res.num_candidates += 1
            minus_upper.obj_val = self.obj(minus_upper.extension)
            minus_upper.bnd_val = self.bnd(minus_upper.extension)
            heappush(self.heap, minus_upper)

            if not curr_node.locked_attrs[j]: # if j is not a locked attribute
                plus_lower = curr_node.get_plus_lower(j)
                #potentially add a check here
                self.res.num_candidates += 1
                plus_lower.obj_val = self.obj(plus_lower.extension)
                plus_lower.bnd_val = self.bnd(plus_lower.extension)
                heappush(self.heap, plus_lower)

    def push_shift(self, curr_node):
        shifted_node = copy.deepcopy(curr_node)
        shifted_node.active_attr -= 1
        shifted_node.shifted = True
        heappush(self.heap, shifted_node)

    def run(self):

        root_ext = np.arange(self.ctx.n)
        max_bnd = self.bnd(root_ext)
        root_node = IPNode(self.ctx, self.ctx.closure(root_ext), self.ctx.m-1, np.zeros(self.ctx.m, dtype=int))
        root_node.obj_val = self.obj(root_ext)
        root_node.bnd_val = max_bnd

        heappush(self.heap, root_node)
        
        while self.heap: # while queue is not empty

            curr_node = heappop(self.heap)
            j = curr_node.active_attr

            if not curr_node.shifted:
                curr_node.obj_val = self.obj(curr_node.extension)
                curr_node.bnd_val = self.bnd(curr_node.extension)
                
                # self.res.max_obj = max(self.curr_node.obj_val, self.res.max_obj)
                if curr_node.obj_val >= self.res.max_obj:
                        self.res.max_obj = curr_node.obj_val
                        self.res.opt_node = curr_node
                if curr_node.bnd_val > self.res.max_obj: # or self.curr_node == self.root: # root node check because bnd > max_obj check fails on root
                    closed_intent = self.ctx.closure(curr_node.extension) # maybe make this conditional to avoid repeat closure
                    if U.is_canonical(curr_node.intent, closed_intent, j):
                        curr_node.intent = closed_intent
                        self.res.num_nodes += 1
                        if self.res.max_obj == max_bnd:
                            break
                        self.push_children(curr_node)
                        if j>0:
                            self.push_shift(curr_node)
            else:
                self.push_children(curr_node)
                if j>0:
                    self.push_shift(curr_node)

class Search3(IPSearch): # best first search
    def __init__(self, heap, ctx, res, obj, bnd):
        super().__init__( heap, ctx, res, obj, bnd)

    def run(self):
        root_ext = np.arange(self.ctx.n)
        max_bnd = self.bnd(root_ext)
        root_node = IPNode(self.ctx, self.ctx.closure(root_ext), self.ctx.m-1, np.zeros(self.ctx.m, dtype=int))
        root_node.obj_val = self.obj(root_ext)
        root_node.bnd_val = max_bnd

        heappush(self.heap, root_node)
        
        while self.heap: # while queue is not empty

            curr_node = heappop(self.heap)
            curr_node.obj_val = self.obj(curr_node.extension)
            curr_node.bnd_val = self.bnd(curr_node.extension)
            
            j = curr_node.active_attr
            # self.res.max_obj = max(self.curr_node.obj_val, self.res.max_obj)
            if curr_node.obj_val >= self.res.max_obj:
                    self.res.max_obj = curr_node.obj_val
                    self.res.opt_node = curr_node
            if curr_node.bnd_val > self.res.max_obj: # or self.curr_node == self.root: # root node check because bnd > max_obj check fails on root
                closed_intent = self.ctx.closure(curr_node.extension) # maybe make this conditional to avoid repeat closure
                if U.is_canonical(curr_node.intent, closed_intent, j):
                    curr_node.intent = closed_intent
                    self.res.num_nodes += 1
                    if self.res.max_obj == max_bnd:
                        break
                    self.push_children(curr_node, j)
                    if j>0:
                        self.push_children(curr_node, j-1)