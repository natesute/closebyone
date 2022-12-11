from iterative_cbo import BFS
from recursive_cbo import DFS
from search import Results, Context, Utilities as U, Extent, Node, NodeBFS
import numpy as np


if __name__ == "__main__":
    # target = U.rand_target_col(10, 0.5, 0)
    # objects = U.rand_disc_num_array(10, 5)
    
    target = np.array([1,0,1,0,1,0,0,1,0,1])
    objects = np.array([[5, 7, 3, 7, 4],
                        [7, 3, 7, 4, 7],
                        [3, 1, 2, 8, 8],
                        [6, 4, 8, 4, 1],
                        [6, 2, 8, 5, 9],
                        [6, 4, 4, 5, 5],
                        [2, 6, 2, 8, 6],
                        [2, 5, 5, 4, 1],
                        [7, 3, 7, 2, 4],
                        [2, 3, 2, 3, 9]])
    # target = np.array([0,1,1,0,0,0,1,1,1,1])
    # objects = np.array([[4,1], [7,10], [5,5], [2,2], [1,1], [3,3], [6,6], [8,8], [9,9], [10,10]])
    print(target)
    print(objects)

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

    root = Node(root_ext, 0, float("inf"), m-1, [0]*m)

    my_search = BFS(NodeBFS(root), [], context, Results())

    my_search.run(root)

    print("BFS results:\n\n")
    print(my_search.res)

    my_search = DFS(root, [], context, Results())

    my_search.run_numerical(root.extent)

    print("DFS results:\n\n")
    print(my_search.res)