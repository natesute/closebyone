import numpy as np
import matplotlib.pyplot as plt
from main import rand_target_col, rand_disc_num_array, disc_num_to_bin, run_baseline, run_cbo

def run_test(i, j, a):
    nodes_cbo = []
    nodes_bin = []
    time_cbo = []
    time_bin = []
    edges_cbo = []
    edges_bin = []
    loops = 15
    for _ in range(loops):
        target = rand_target_col(i,a)
        data_cbo = rand_disc_num_array(i, j)
        data_bin = disc_num_to_bin(data_cbo)

        time_cbo.append(run_cbo(data_cbo, target).time)
        time_bin.append(run_baseline(data_bin, target).time)
        edges_cbo.append(run_cbo(data_cbo, target).num_edges)
        edges_bin.append(run_baseline(data_bin, target).num_edges)
        nodes_cbo.append(run_cbo(data_cbo, target).num_nodes)
        nodes_bin.append(run_baseline(data_bin, target).num_nodes)
        
    time_cbo_result = float(np.average(np.array(time_cbo)))
    time_bin_result = float(np.average(np.array(time_bin)))
    edges_cbo_result = float(np.average(np.array(edges_cbo)))
    edges_bin_result = float(np.average(np.array(edges_bin)))
    nodes_cbo_result = float(np.average(np.array(nodes_cbo)))
    nodes_bin_result = float(np.average(np.array(nodes_bin)))
    
    return time_cbo_result, time_bin_result, edges_cbo_result, edges_bin_result, nodes_cbo_result, nodes_bin_result

def get_results(ms, ns, alphas):
    time_cbo = {}
    time_bin = {}
    edges_cbo = {}
    edges_bin = {}
    nodes_cbo = {}
    nodes_bin = {}
    for m in ms:
        for n in ns:
            # either ms[-1] or n[-1] will be used to be the static parameter, others can be ignored
            if m != ms[-1] and n != ns[-1]:
                continue
            for a in alphas:
                time_cbo[m,n,a], time_bin[m,n,a], edges_cbo[m,n,a], edges_bin[m,n,a], nodes_cbo[m,n,a], nodes_bin[m,n,a] = run_test(m,n,a)
    return time_cbo, time_bin, edges_cbo, edges_bin, nodes_cbo, nodes_bin

def display_plots(ms, ns, alphas, time_cbo, time_bin, edges_cbo, edges_bin, nodes_cbo, nodes_bin):
    # displays times
    plt.subplots(3, 3, figsize=(10, 5))

    res_cbo = [time_cbo[m, ns[-1], alphas[-1]] for m in ms]
    res_bin = [time_bin[m, ns[-1], alphas[-1]] for m in ms]

    plt.subplot(3, 3, 1)
    plt.plot(ms, res_cbo, marker='o', label='interval patterns')
    plt.plot(ms, res_bin, marker='s', label='binarised')
    if len(ms) > 1:
        plt.xlim(ms[0], ms[-1])
    plt.ylabel('time (s)')
    plt.xlabel('# data')
    plt.legend()

    res_cbo = [time_cbo[ms[-1], n, alphas[-1]] for n in ns]
    res_bin = [time_bin[ms[-1], n, alphas[-1]] for n in ns]

    plt.subplot(3, 3, 2)
    plt.plot(ns, res_cbo, marker='o', label='interval patterns')
    plt.plot(ns, res_bin, marker='s', label='binarised')
    if len(ns) > 1:
        plt.xlim(ns[0], ns[-1])
    plt.ylabel('time (s)')
    plt.xlabel('# features')
    plt.legend()
    
    res_cbo = [time_cbo[ms[-1], ns[-1], a] for a in alphas]
    res_bin = [time_bin[ms[-1], ns[-1], a] for a in alphas]

    plt.subplot(3, 3, 3)
    plt.plot(alphas, res_cbo, marker='o', label='interval patterns')
    plt.plot(alphas, res_bin, marker='s', label='binarised')
    if len(alphas) > 1:
        plt.xlim(alphas[0], alphas[-1])
    plt.ylabel('time (s)')
    plt.xlabel('% positive target')
    plt.legend()

    # display edges

    res_cbo = [edges_cbo[m, ns[-1], alphas[-1]] for m in ms]
    res_bin = [edges_bin[m, ns[-1], alphas[-1]] for m in ms]

    plt.subplot(3, 3, 4)
    plt.plot(ms, res_cbo, marker='o', label='interval patterns')
    plt.plot(ms, res_bin, marker='s', label='binarised')
    if len(ms) > 1:
        plt.xlim(ms[0], ms[-1])
    plt.ylabel('# edges')
    plt.xlabel('# data')
    plt.legend()

    res_cbo = [edges_cbo[ms[-1], n, alphas[-1]] for n in ns]
    res_bin = [edges_bin[ms[-1], n, alphas[-1]] for n in ns]

    plt.subplot(3, 3, 5)
    plt.plot(ns, res_cbo, marker='o', label='interval patterns')
    plt.plot(ns, res_bin, marker='s', label='binarised')
    if len(ns) > 1:
        plt.xlim(ns[0], ns[-1])
    plt.ylabel('# edges')
    plt.xlabel('# features')
    plt.legend()
    

    res_cbo = [edges_cbo[ms[-1], ns[-1], a] for a in alphas]
    res_bin = [edges_bin[ms[-1], ns[-1], a] for a in alphas]

    plt.subplot(3, 3, 6)
    plt.plot(alphas, res_cbo, marker='o', label='interval patterns')
    plt.plot(alphas, res_bin, marker='s', label='binarised')
    if len(alphas) > 1:
        plt.xlim(alphas[0], alphas[-1])
    plt.ylabel('# edges')
    plt.xlabel('% positive target')
    plt.legend()

    # display nodes

    res_cbo = [nodes_cbo[m, ns[-1], alphas[-1]] for m in ms]
    res_bin = [nodes_bin[m, ns[-1], alphas[-1]] for m in ms]

    plt.subplot(3, 3, 7)
    plt.plot(ms, res_cbo, marker='o', label='interval patterns')
    plt.plot(ms, res_bin, marker='s', label='binarised')
    if len(ms) > 1:
        plt.xlim(ms[0], ms[-1])
    plt.ylabel('# nodes')
    plt.xlabel('# data')
    plt.legend()

    res_cbo = [nodes_cbo[ms[-1], n, alphas[-1]] for n in ns]
    res_bin = [nodes_bin[ms[-1], n, alphas[-1]] for n in ns]

    plt.subplot(3, 3, 8)
    plt.plot(ns, res_cbo, marker='o', label='interval patterns')
    plt.plot(ns, res_bin, marker='s', label='binarised')
    if len(ns) > 1:
        plt.xlim(ns[0], ns[-1])
    plt.ylabel('# nodes')
    plt.xlabel('# features')
    plt.legend()

    # default static parameters to largest value
    res_cbo = [nodes_cbo[ms[-1], ns[-1], a] for a in alphas]
    res_bin = [nodes_bin[ms[-1], ns[-1], a] for a in alphas]

    plt.subplot(3, 3, 9)
    plt.plot(alphas, res_cbo, marker='o', label='interval patterns')
    plt.plot(alphas, res_bin, marker='s', label='binarised')
    if len(alphas) > 1:
        plt.xlim(alphas[0], alphas[-1])
    plt.ylabel('# nodes')
    plt.xlabel('% positive target')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    ms = [5,6,7,8,9,10,11]
    ns = [2,3,4,5,6]
    alphas = [0.2,0.4,0.6,0.8]
    time_cbo, time_bin, edges_cbo, edges_bin, nodes_cbo, nodes_bin = get_results(ms, ns, alphas)
    display_plots(ms, ns, alphas, time_cbo, time_bin, edges_cbo, edges_bin, nodes_cbo, nodes_bin)