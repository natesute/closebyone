from best_first_cbo import Context, Utilities, Extent, Node, BFS
import numpy as np


if __name__ == "__main__":
    #target = Utilities.rand_target_col(10, 0.5)
    #objects = Utilities.rand_disc_num_array(10, 4, 0)
    objects = np.array([[2,1],
                     [3,5]])
    target = np.array([0,1])
    m = len(objects[0])
    n = len(objects)
    root_ext = Extent(np.arange(n), objects)
    target_mean_root = Utilities.target_mean(root_ext)
    obj = Utilities.impact_obj(target)
    bnd = Utilities.impact_bnd(target)
    intent = root_ext.get_closure()

    root = Node(root_ext, intent, obj(root_ext.indices), bnd(root_ext.indices), [0]*m, m-1)
    context = Context(target, objects, obj, bnd)

    my_bfs = BFS(root, np.empty, context)

    my_bfs.run(root)