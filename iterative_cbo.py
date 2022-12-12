from heapq import heappop, heappush
import numpy as np
from search import IPSearch, Utilities as U


class BFS(IPSearch): # best first search
    def __init__(self, root, curr_node, heap, context, res):
        super().__init__(root, curr_node, heap, context, res)


    def push_children(self, j):
        if not self.curr_node.intent.fully_closed(j):
            new_locked_attrs = np.copy(self.curr_node.locked_attrs)
            new_locked_attrs[j] = 1
            minus_upper = self.curr_node.get_minus_upper(j)

            self.res.num_candidates += 1
            heappush(self.heap, minus_upper)

            if not self.curr_node.locked_attrs[j]: # if j is not a locked attribute
                plus_lower = self.curr_node.get_plus_lower(j)
                #potentially add a check here
                self.res.num_candidates += 1
                heappush(self.heap, plus_lower)

    def search(self):
        max_bnd = self.context.bnd(self.root.extent.indices)
        heappush(self.heap, self.root)
        
        while self.heap: # while queue is not empty]

            self.curr_node = heappop(self.heap)
            
            j = self.curr_node.active_attr
            # self.res.max_obj = max(self.curr_node.obj_val, self.res.max_obj)
            if self.curr_node.obj_val >= self.res.max_obj:
                    self.res.max_obj = self.curr_node.obj_val
                    self.res.best_node = self.curr_node
            if self.context.bnd(self.curr_node.extent.indices) > self.res.max_obj or self.curr_node == self.root: # root node check because obj > max_obj check fails on root
                closed_intent = self.curr_node.extent.get_closure()
                if U.is_canonical(self.curr_node.intent, closed_intent, j):
                    self.curr_node.intent = closed_intent
                    self.res.num_nodes += 1
                    if self.res.max_obj == max_bnd:
                        break
                    self.push_children(j)
                    if j>0:
                        self.push_children(j-1)

    def run(self):
        return self.search()