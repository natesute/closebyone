import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from recursive_cbo import DFS
from search import Results, Context, Utilities as U, Extent, Node, Search

def run_test(i, j, a):
    nodes_num = []
    nodes_bin = []
    time_num = []
    time_bin = []
    edges_num = []
    edges_bin = []
    loops = 2
    for _ in range(loops):
        target = U.rand_target_col(i,a)
        objects_num = U.rand_disc_num_array(i, j)
        objects_bin = U.disc_num_to_bin(objects_num)

        time_num.append(DFS.res_numerical(objects_num, target).time)
        time_bin.append(DFS.res_binarised(objects_bin, target).time)
        edges_num.append(DFS.res_numerical(objects_num, target).num_edges)
        edges_bin.append(DFS.res_binarised(objects_bin, target).num_edges)
        nodes_num.append(DFS.res_numerical(objects_num, target).num_nodes)
        nodes_bin.append(DFS.res_binarised(objects_bin, target).num_nodes)
        
    time_num_result = float(np.average(np.array(time_num)))
    time_bin_result = float(np.average(np.array(time_bin)))
    edges_num_result = float(np.average(np.array(edges_num)))
    edges_bin_result = float(np.average(np.array(edges_bin)))
    nodes_num_result = float(np.average(np.array(nodes_num)))
    nodes_bin_result = float(np.average(np.array(nodes_bin)))
    
    return time_num_result, time_bin_result, edges_num_result, edges_bin_result, nodes_num_result, nodes_bin_result

def get_results(ms, ns, alphas):
    time_num = {}
    time_bin = {}
    edges_num = {}
    edges_bin = {}
    nodes_num = {}
    nodes_bin = {}
    for m in ms:
        for n in ns:
            # either ms[-1] or n[-1] will be used to be the static parameter, others can be ignored
            if m != ms[-1] and n != ns[-1]:
                continue
            for a in alphas:
                time_num[m,n,a], time_bin[m,n,a], edges_num[m,n,a], edges_bin[m,n,a], nodes_num[m,n,a], nodes_bin[m,n,a] = run_test(m,n,a)
    return time_num, time_bin, edges_num, edges_bin, nodes_num, nodes_bin

def display_plots(ms, ns, alphas, time_num, time_bin, edges_num, edges_bin, nodes_num, nodes_bin):
    # displays times
    plt.subplots(3, 3, figsize=(10, 5))

    res_num = [time_num[m, ns[-1], alphas[-1]] for m in ms]
    res_bin = [time_bin[m, ns[-1], alphas[-1]] for m in ms]

    plt.subplot(3, 3, 1)
    plt.plot(ms, res_num, marker='o', label='interval patterns')
    plt.plot(ms, res_bin, marker='s', label='binarised')
    if len(ms) > 1:
        plt.xlim(ms[0], ms[-1])
    plt.ylabel('time (s)')
    plt.xlabel('# data')
    plt.legend()

    res_num = [time_num[ms[-1], n, alphas[-1]] for n in ns]
    res_bin = [time_bin[ms[-1], n, alphas[-1]] for n in ns]

    plt.subplot(3, 3, 2)
    plt.plot(ns, res_num, marker='o', label='interval patterns')
    plt.plot(ns, res_bin, marker='s', label='binarised')
    if len(ns) > 1:
        plt.xlim(ns[0], ns[-1])
    plt.ylabel('time (s)')
    plt.xlabel('# features')
    plt.legend()
    
    res_num = [time_num[ms[-1], ns[-1], a] for a in alphas]
    res_bin = [time_bin[ms[-1], ns[-1], a] for a in alphas]

    plt.subplot(3, 3, 3)
    plt.plot(alphas, res_num, marker='o', label='interval patterns')
    plt.plot(alphas, res_bin, marker='s', label='binarised')
    if len(alphas) > 1:
        plt.xlim(alphas[0], alphas[-1])
    plt.ylabel('time (s)')
    plt.xlabel('% positive target')
    plt.legend()

    # display edges

    res_num = [edges_num[m, ns[-1], alphas[-1]] for m in ms]
    res_bin = [edges_bin[m, ns[-1], alphas[-1]] for m in ms]

    plt.subplot(3, 3, 4)
    plt.plot(ms, res_num, marker='o', label='interval patterns')
    plt.plot(ms, res_bin, marker='s', label='binarised')
    if len(ms) > 1:
        plt.xlim(ms[0], ms[-1])
    plt.ylabel('# edges')
    plt.xlabel('# data')
    plt.legend()

    res_num = [edges_num[ms[-1], n, alphas[-1]] for n in ns]
    res_bin = [edges_bin[ms[-1], n, alphas[-1]] for n in ns]

    plt.subplot(3, 3, 5)
    plt.plot(ns, res_num, marker='o', label='interval patterns')
    plt.plot(ns, res_bin, marker='s', label='binarised')
    if len(ns) > 1:
        plt.xlim(ns[0], ns[-1])
    plt.ylabel('# edges')
    plt.xlabel('# features')
    plt.legend()
    

    res_num = [edges_num[ms[-1], ns[-1], a] for a in alphas]
    res_bin = [edges_bin[ms[-1], ns[-1], a] for a in alphas]

    plt.subplot(3, 3, 6)
    plt.plot(alphas, res_num, marker='o', label='interval patterns')
    plt.plot(alphas, res_bin, marker='s', label='binarised')
    if len(alphas) > 1:
        plt.xlim(alphas[0], alphas[-1])
    plt.ylabel('# edges')
    plt.xlabel('% positive target')
    plt.legend()

    # display nodes

    res_num = [nodes_num[m, ns[-1], alphas[-1]] for m in ms]
    res_bin = [nodes_bin[m, ns[-1], alphas[-1]] for m in ms]

    plt.subplot(3, 3, 7)
    plt.plot(ms, res_num, marker='o', label='interval patterns')
    plt.plot(ms, res_bin, marker='s', label='binarised')
    if len(ms) > 1:
        plt.xlim(ms[0], ms[-1])
    plt.ylabel('# nodes')
    plt.xlabel('# data')
    plt.legend()

    res_num = [nodes_num[ms[-1], n, alphas[-1]] for n in ns]
    res_bin = [nodes_bin[ms[-1], n, alphas[-1]] for n in ns]

    plt.subplot(3, 3, 8)
    plt.plot(ns, res_num, marker='o', label='interval patterns')
    plt.plot(ns, res_bin, marker='s', label='binarised')
    if len(ns) > 1:
        plt.xlim(ns[0], ns[-1])
    plt.ylabel('# nodes')
    plt.xlabel('# features')
    plt.legend()

    # default static parameters to largest value
    res_num = [nodes_num[ms[-1], ns[-1], a] for a in alphas]
    res_bin = [nodes_bin[ms[-1], ns[-1], a] for a in alphas]

    plt.subplot(3, 3, 9)
    plt.plot(alphas, res_num, marker='o', label='interval patterns')
    plt.plot(alphas, res_bin, marker='s', label='binarised')
    if len(alphas) > 1:
        plt.xlim(alphas[0], alphas[-1])
    plt.ylabel('# nodes')
    plt.xlabel('% positive target')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    # ms = [5,6,7,8,9,10,11]
    # ns = [2,3,4,5,6]
    # alphas = [0.2,0.4,0.6,0.8]
    # time_num, time_bin, edges_num, edges_bin, nodes_num, nodes_bin = get_results(ms, ns, alphas)
    # display_plots(ms, ns, alphas, time_num, time_bin, edges_num, edges_bin, nodes_num, nodes_bin)
    # display_plots(*get_results(ms, ns, alphas))
    
    target = U.rand_target_col(10, 0.5, 0)
    objects = U.rand_disc_num_array(10, 4)
    # target = np.array([1,0])
    # objects = np.array([[1,2], [3,4]])
    obj = U.impact_obj(target)
    bnd = U.impact_bnd(target)
    get_target_mean = U.target_mean(target)
    context = Context(target, objects, obj, bnd)
    m = len(objects[0])
    n = len(objects)
    root_ext = Extent(np.arange(n), objects)
    target_mean_root = get_target_mean(root_ext.indices)

    intent = root_ext.get_closure()

    root = Node(root_ext, intent, obj(root_ext.indices), bnd(root_ext.indices), [0]*m, m-1)

    my_search = DFS(root, [], context, Results())

    my_search.run_numerical(root)

    print(my_search.res)