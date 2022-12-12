from search import Results
import numpy as np
import timeit
from search import PropSearch, Extent, PropNode
from copy import deepcopy

# binarised approach

class PropSearch:

    def __init__(self, curr_node, context, res):
        self.curr_node = curr_node
        self.context = context
        self.res = Results()

    def run(self):
        return self.search(0)

    def search(self, i):
        self.res.num_nodes += 1
        num = len(self.curr_node.query)

        for prop in range(i, num):
            self.res.num_candidates += 1
            if self.res.max_obj == self.res.max_bnd:
                return
            aug_indices = self.curr_node.extent[self.context.objects[self.curr_node.extent, prop]] 
            aug_extent = Extent(aug_indices, self.context.objects)
            if len(aug_extent) == 0:
                continue
            # proposition is already implied
            if self.curr_node.query[prop] == True:
                continue
            self.res.num_candidates += 1 # should i be doing this twice?
            aug_query = deepcopy(self.curr_node.query)
            aug_query[prop] = True
            aug_node = PropNode(self.context, aug_query, aug_extent)

            # self.res.max_obj = max(self.context.obj(aug_extent), self.res.max_obj)
            if self.context.obj(aug_extent.indices) >= self.res.max_obj:
                self.res.max_obj = self.context.obj(aug_extent.indices)
                self.res.best_node = aug_node

            if self.res.max_obj > self.context.bnd(aug_extent.indices):
                continue

            prefix_pres = True
            for k in range(prop):
                if self.curr_node.query[k]:
                    pass
                elif len(aug_extent) <= self.context.ext_sizes[k] and self.context.implied_on(k, aug_extent.indices):
                    prefix_pres = False
                    break
            if not prefix_pres:
                continue

            for k in range(prop + 1, num):
                if len(aug_extent) <= self.context.ext_sizes[k] and self.context.implied_on(k, aug_extent.indices):
                    aug_query[k] = True
            self.search(prop + 1)