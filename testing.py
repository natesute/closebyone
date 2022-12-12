from iterative_cbo import BFS
from recursive_cbo import DFS
from binarised import PropSearch

from search import Results, Context, Utilities as U, Extent, IPNode, PropNode, Query
import numpy as np
import timeit

class Test:
    def __init__(self, objects, target, search_type):
        self.objects = objects
        self.target = target
        self.search_type = search_type

    def run(self):
        obj = U.impact_obj(self.target)
        bnd = U.impact_bnd(self.target)
        context = Context(self.target, self.objects, obj, bnd)
        m = len(self.objects[0])
        n = len(self.objects)

        root_ext = Extent(np.arange(n), self.objects)
        

        if self.search_type == "binarised":
            root_ext = Extent(np.arange(n), self.objects)
            root_query = Query(context.check_root())
            root = PropNode(context, root_query, root_ext)
            context.ext_sizes = np.add.reduce(context.objects)
            my_search = PropSearch(root, context, Results())
        elif self.search_type == "bfs":
            root_ext = Extent(np.arange(n), self.objects)
            intent = root_ext.get_closure()
            root = IPNode(context, intent, m-1, [0]*m)
            my_search = BFS(root, root, [], context, Results())
        elif self.search_type == "dfs":
            root_ext = Extent(np.arange(n), self.objects)
            intent = root_ext.get_closure()
            root = IPNode(context, intent, m-1, [0]*m)
            my_search = DFS(root, root, [], context, Results())
        my_search.res.time = timeit.timeit(my_search.run, number=1)
        return my_search.res
    

if __name__ == "__main__":
    target = U.rand_target_col(20, 0.5, 0)
    objects = U.rand_disc_num_array(20, 6)
    objects_bin = U.disc_num_to_bin(objects) # binarise objects
    # test = Test(objects, target, "dfs")
    # print("dfs\n\n")
    # print(test.run())
    test = Test(objects, target, "bfs")
    print("##############################\n          BFS\n")
    print(test.run())
    # test = Test(objects_bin, target, "binarised")
    # print("##############################\n          BINARISED\n\n")
    # print(test.run())
    test = Test(objects, target, "dfs")
    print("##############################\n          DFS\n")
    print(test.run())