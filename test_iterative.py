from iterative_cbo import BFS
from search import Results, Context, Utilities as U, Extent, Node
import numpy as np


if __name__ == "__main__":
    target = U.rand_target_col(10, 0.5, 0)
    objects = U.rand_disc_num_array(10, 4)
    # target = np.array([1,0])
    # objects = np.array([[1,2], [3,4]])

    obj = U.impact_obj(target)
    bnd = U.impact_bnd(target)
    get_target_mean = U.target_mean(target)
    context = Context(target, objects, obj, bnd)
    #objects = np.array([[2,1],[3,5]])
    #target = np.array([0,1])
    m = len(objects[0])
    n = len(objects)
    root_ext = Extent(np.arange(n), objects)
    target_mean_root = get_target_mean(root_ext.indices)
    
    intent = root_ext.get_closure()

    root = Node(context, intent, m-1, [0]*m)

    my_search = BFS(Node(root), [], context, Results())

    my_search.run(root)

    print(my_search.res)