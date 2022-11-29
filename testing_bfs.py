from best_first_cbo import Intent, Extent, Node, CloseByOneBFS
import numpy as np

RNG = np.random.default_rng(seed=0)

# create random target column
def rand_target_col(rows, alpha):
    target_col = np.zeros((rows, 1))
    end_of_1s = int(len(target_col) * alpha)
    for i in range(end_of_1s):
        target_col[i] = 1
    RNG.shuffle(target_col)
    return target_col

# create random discrete numerical array
def rand_disc_num_array(rows, cols):
    data_cols = np.random.randint(1, 10, (rows, cols))
    return data_cols

if __name__ == "__main__":
    data = np.array([[2,1],
                     [3,5]])
    target = np.array([1,0])
    m = len(data[0])
    n = len(data)
    root_extent = Extent(np.arange(len(data)), data)

    targ_perc_root = target[root_extent.indices].mean()

    obj = lambda ext : (len(ext.indices) / n) * (target[ext.indices].mean() - targ_perc_root) # impact
    bnd = lambda ext : (target[ext.indices].sum() / n) * (1 - targ_perc_root)

    target = rand_target_col(10, 0.5)
    data = rand_disc_num_array(10, 4)
    bfs = CloseByOneBFS(target, data, obj, bnd)
    intent = root_extent.get_closure()
    root = Node(root_extent, intent, bfs.f(root_extent), bfs.g(root_extent), [0]*m, m-1)
    bfs.run(root)