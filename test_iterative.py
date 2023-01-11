from iterative_cbo import BFS
from search import Results, Context, Utilities as U, Extent, Node
import numpy as np


if __name__ == "__main__":
    labels = U.rand_labels(10, 0.5, 0)
    objects = U.rand_disc_num_array(10, 4)
    # labels = np.array([1,0])
    # objects = np.array([[1,2], [3,4]])

    obj = U.impact_obj
    bnd = U.impact_bnd
    context = Context(labels, objects, obj, bnd)
    #objects = np.array([[2,1],[3,5]])
    #labels = np.array([0,1])
    m = len(objects[0])
    n = len(objects)
    root_ext = Extent(np.arange(n), objects)
    
    intent = root_ext.get_closure()

    root = Node(context, intent, m-1, [0]*m)

    my_search = BFS(Node(root), [], context, Results())

    my_search.run(root)

    print(my_search.res)