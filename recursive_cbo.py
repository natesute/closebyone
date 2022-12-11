import timeit # type: ignore
import numpy as np # type: ignore
from numba import int64, float64 # type: ignore
from search import Search, Results, Context, Utilities as U, Extent, Intent, Node
from numba.core.types import Array # type: ignore
import copy # type: ignore
# from numba import njit
# from numba.experimental import jitclass

class DFS(Search):

    def __init__(self, curr_node, heap, context, res):
        super().__init__(curr_node, heap, context, res)

    def search_plus_lower(self, intent: Intent, extent: Extent, j):
        # raise lower bound of j
        self.res.num_candidates += 1
        extent = self.get_extent(intent)
        # curr_node = Node(extent, intent, 0, 0, None, j)
        # is curr_node needed

        if len(extent) > 0:
            # self.res.max_obj = max(self.context.obj(extent.indices), self.res.max_obj)
            if self.context.obj(extent.indices) >= self.res.max_obj:
                    self.res.max_obj = self.context.obj(extent.indices)
                    self.res.best_query = intent
            if self.res.max_obj == self.res.max_bnd:
                self.res.num_nodes += 1
                return

            if self.context.bnd(extent.indices) >= self.res.max_obj:
                new_intent = extent.get_closure()
                
                # check if new intent is canonical
                if U.is_canonical(intent, new_intent, j):
                    intent = new_intent
                    self.res.num_nodes += 1
                    print(intent)
                    
                    # print(intent)
                    # check if bounds can be further changed on j
                    if not intent.fully_closed(j):
                        # branch current attribute, both bounds
                        self.search_minus_upper(intent.get_minus_upper(j), copy.deepcopy(extent), j)
                        
                        self.search_plus_lower(intent.get_plus_lower(j), copy.deepcopy(extent), j)

                    if j > 0:
                        self.search(copy.deepcopy(intent), copy.deepcopy(extent), j - 1)


    def search_minus_upper(self, intent: Intent, extent: Extent, j):
        # lower upper bound of j
        self.res.num_candidates += 1
        extent = self.get_extent(copy.deepcopy(intent))

        if len(extent) > 0:
            # self.res.max_obj = max(self.context.obj(extent.indices), self.res.max_obj)
            if self.context.obj(extent.indices) >= self.res.max_obj:
                    
                    self.res.max_obj = self.context.obj(extent.indices)
                    self.res.best_query = intent
                    self.res.num_nodes += 1
                    # check if maximum bound has been hit
                    if self.res.max_obj == self.res.max_bnd:
                        return
            
            if self.context.bnd(extent.indices) > self.res.max_obj:
                new_intent = extent.get_closure()
                if U.is_canonical(intent, new_intent, j):             
                    intent = new_intent                
                    
                    self.res.num_nodes += 1
                    print(intent)
                    # check if bounds can be further changed on j
                    if not intent.fully_closed(j):
                        # branch current attribute, only upper bound
                        self.search_minus_upper(intent.get_minus_upper(j), copy.deepcopy(extent), j)
                        

                    if j > 0:  # because j==0 is value col
                        self.search(copy.deepcopy(intent), copy.deepcopy(extent), j - 1)

    # search changes on jth attribute downwards
    def search(self, intent: Intent, extent: Extent, j):
    #     print("next attribute down\n")
        if not intent.fully_closed(j): # is this needed here? (may be needed if attributes begin closed)
            # search aug children of minus-ing upper bound
            self.search_minus_upper(intent.get_minus_upper(j), copy.deepcopy(extent), j)

            # search aug children of plus-ing lower bound
            self.search_plus_lower(intent.get_plus_lower(j), copy.deepcopy(extent), j)

        if j > 0:
            # search changes on j-1th attribute downwards
            self.search(copy.deepcopy(intent), copy.deepcopy(extent), j - 1)

    def run_numerical(self, extent):
        self.res.targ_mean_root = self.context.get_target_mean(extent.indices)
        self.res.max_bnd = self.context.bnd(extent.indices)
        intent = extent.get_closure()
        # self.res.max_obj = max(self.context.obj(extent.indices), self.res.max_obj)
        if self.context.obj(extent.indices) >= self.res.max_obj:
                self.res.max_obj = self.context.obj(extent.indices)
                self.res.best_query = intent
        
        self.res.num_nodes += 1
        
        # check if inital objective value is maximum bound
        if self.res.max_obj == self.res.max_bnd:
            return

        # starts searching from last attribute and decrements
        j = self.context.m - 1
        def closure():
            return self.search(copy.deepcopy(intent), copy.deepcopy(extent), j)
        self.res.time = timeit.timeit(closure, number=1)
        return

    # binarised approach

    def binarised(self, ext_sizes, query, extent, i):
        self.res.num_nodes += 1
        num = len(query)

        for prop in range(i, num):
            self.res.num_candidates += 1
            if self.res.max_obj == self.res.max_bnd:
                return
            aug_extent = extent[self.context.objects[extent, prop]]
            if aug_extent.size == 0:
                continue
            # proposition is already implied
            if query[prop] == True:
                continue
            self.res.num_candidates += 1 # should i be doing this twice?
            aug_query = query.copy()
            aug_query[prop] = True

            # self.res.max_obj = max(self.context.obj(aug_extent), self.res.max_obj)
            if self.context.obj(aug_extent) >= self.res.max_obj:
                self.res.max_obj = self.context.obj(aug_extent)
                self.res.best_query = aug_query

            if self.res.max_obj > self.context.bnd(aug_extent.indices):
                continue

            prefix_pres = True
            for k in range(prop):
                if query[k]:
                    pass
                elif len(aug_extent) <= ext_sizes[k] and Search.implied_on(k, aug_extent):
                    prefix_pres = False
                    break
            if not prefix_pres:
                continue

            for k in range(prop + 1, num):
                if len(aug_extent) <= ext_sizes[k] and Search.implied_on(k, aug_extent):
                    aug_query[k] = True
            self.binarised(ext_sizes, aug_query, aug_extent, prop + 1)


    def run_binarised(self):
        root_query = self.root(self.context.objects)
        extent = np.arange(len(self.context.objects))
        ext_sizes = np.add.reduce(self.context.objects)
        self.context.target_mean_root = self.context.target_mean(extent.indices)
        self.res.max_bnd = self.res.bnd(extent.indices)

        def closure():
            return self.binarised(ext_sizes, root_query, extent, 0)
        self.res.time = timeit.timeit(closure, number=1)
        return