import timeit # type: ignore
import numpy as np # type: ignore
from numba import int64, float64 # type: ignore
from search import IPSearch, Utilities as U
from numba.core.types import Array # type: ignore
import copy # type: ignore
# from numba import njit
# from numba.experimental import jitclass

class DFS(IPSearch):

    def __init__(self, root, curr_node, heap, context, res):
        super().__init__(root, curr_node, heap, context, res)

    def search(self, node=None):
        if node is None:
            node = self.root
        j = node.active_attr
        # lower upper bound of j
        self.res.num_candidates += 1
        extent = node.extent
        intent = extent.get_closure()

        if len(extent) > 0:
            # self.res.max_obj = max(self.context.obj(extent.indices), self.res.max_obj)
            if node.obj_val >= self.res.max_obj:
                    
                    self.res.max_obj = node.obj_val
                    self.res.best_node = node
                    self.res.num_nodes += 1
                    # check if maximum bound has been hit
                    if self.res.max_obj == self.context.max_bnd:
                        return
            
            if node.bnd_val > self.res.max_obj:
                new_intent = extent.get_closure()
                if U.is_canonical(intent, new_intent, j):             
                    intent = new_intent                
                    
                    self.res.num_nodes += 1
                    # check if bounds can be further changed on j
                    if not node.intent.fully_closed(j):
                        # branch current attribute, only upper bound
                        self.search(node.get_minus_upper(j))
                        
                        if not node.locked_attrs[j]:
                            self.search(node.get_plus_lower(j))
                    if node.active_attr > 0:
                        node.active_attr = j - 1
                        self.search(copy.deepcopy(node))

    def run(self):
        return self.search()