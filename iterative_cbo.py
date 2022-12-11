from heapq import heappop, heappush
import numpy as np
from search import Search, NodeBFS, Utilities as U


class BFS(Search): # best first search
    def __init__(self, curr_node, heap, context, res):
        super().__init__(curr_node, heap, context, res)


    def push_children(self, j):
        if not self.curr_node.intent.fully_closed(j):
            new_locked_attrs = np.copy(self.curr_node.locked_attrs)
            new_locked_attrs[j] = 1
            new_intent = self.curr_node.get_minus_upper(j)

            self.res.num_candidates+=1
            heappush(self.heap, NodeBFS(new_intent, j, new_locked_attrs))

            if not self.curr_node.locked_attrs[j]: # if j is not a locked attribute
                new_intent = self.curr_node.get_plus_lower(j)
                #potentially add a check here
                self.res.num_candidates += 1
                heappush(self.heap, NodeBFS(new_intent, j, self.curr_node.locked_attrs))

    def run(self, root_node):
        max_bnd = self.context.bnd(root_node.extent.indices)
        heappush(self.heap, root_node)
        
        while self.heap: # while queue is not empty
            self.curr_node = heappop(self.heap)
            
            j = self.curr_node.active_attr
            # self.res.max_obj = max(self.curr_node.obj_val, self.res.max_obj)
            if self.curr_node.obj_val >= self.res.max_obj:
                    self.res.max_obj = self.curr_node.obj_val
                    self.res.best_query = self.curr_node.intent
            if self.context.bnd(self.curr_node.extent.indices) > self.res.max_obj or self.curr_node == root_node: # root node check because obj > max_obj check fails on root
                closed_intent = self.curr_node.extent.get_closure()
                if U.is_canonical(self.curr_node.intent, closed_intent, j):
                    self.curr_node.intent = closed_intent
                    self.res.num_nodes += 1
                    if self.res.max_obj == max_bnd:
                        break
                    self.push_children(j)
                    if j>0:
                        self.push_children(j-1)