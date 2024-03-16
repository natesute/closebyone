from iterative_cbo import Search
from realkd.legacy import impact, cov_incr_mean_bound, impact_count_mean
from realkd.search import Context
from bin_framework import BinTreeSearch
import matplotlib.pyplot as plt
from datetime import datetime

from search import Results, NumContext, Utilities as U
import numpy as np
import timeit
import pandas as pd

results_bin = []
results_num = []

def test(n, m, alpha, seed, my_labels=None, my_objects=None):
    labels = U.rand_labels(n, alpha, seed)
    objects = U.rand_disc_num_array(n, m)

    obj = impact(labels)
    bnd = cov_incr_mean_bound(labels, impact_count_mean(labels))

    bin_objects = U.disc_num_to_bin(objects)
    bin_ctx = Context.from_tab(bin_objects)
    num_ctx = NumContext(objects)

    bin_search = BinTreeSearch(bin_ctx, obj, bnd, Results())
    bin_search.res.time = timeit.timeit(bin_search.run, number=1)
    res_bin = {
        'n': n,
        'm': m,
        'alpha': alpha,
        'seed': seed,
        'time': bin_search.res.time,
        'num_nodes': bin_search.res.num_nodes,
        'num_candidates': bin_search.res.num_candidates
    }

    results_bin.append(res_bin)
    
    num_search = Search([], num_ctx, Results(), obj, bnd)
    num_search.res.time = timeit.timeit(num_search.run, number=1)
    res_num = {
        'n': n,
        'm': m,
        'alpha': alpha,
        'seed': seed,
        'time': num_search.res.time,
        'num_nodes': num_search.res.num_nodes,
        'num_candidates': num_search.res.num_candidates
    }

    results_num.append(res_num)

figs = []
max_vals = {"n": 6, "m": 6, "alpha": 0.5}
var_names = {"n": "objects", "m": "attributes", "alpha": f"label {'%'}", "num_nodes": "nodes", "num_candidates": "candidates", "time": "time (s)"}

def plot_results(bin_df, num_df, independent_var, dependent_var, controlled_vars):
    # creating a filter for getting rows where the values of all other variables are at max
    filter_query = ' and '.join([f"{var}=={max_vals[var]}" for var in controlled_vars])
    
    #grouping by independent variable and plotting the mean of dependent variable
    bin_data=bin_df.query(filter_query).groupby(independent_var)[dependent_var].mean()
    num_data=num_df.query(filter_query).groupby(independent_var)[dependent_var].mean()
    figs.append(plt.figure())
    plt.plot(bin_data.index, bin_data.values, label='binary')
    plt.plot(num_data.index, num_data.values, label='interval-pattern')
    plt.xlabel(f'{var_names[independent_var]}')
    plt.ylabel(f'{var_names[dependent_var]}')
    controlled_str = 'Controlled vars ->      ' + '      '.join([f"{var_names[var]}: {max_vals[var]}" for var in controlled_vars])
    font = {'family':'sans-serif','color':'blue','size':8}
    plt.title(controlled_str, fontdict=font)
    plt.legend()

if __name__ == "__main__":
    repetitions = 30
    for n in range(2, max_vals["n"]+1, 1):
        for m in range(2, max_vals["m"]+1, 1):
            for alpha in range(1, int(max_vals["alpha"]*10)+1, 2):
                alpha = alpha/10
                if n == max_vals["n"] or m == max_vals["m"] or alpha == max_vals["alpha"]: # only tests where a controlled var is at max will be shown
                    for _ in range(repetitions):
                        seed = int(datetime.now().timestamp()%1*1000)
                        np.random.seed(seed)
                        # print("n:", n, "m:", m, "alpha:", alpha, "seed:", seed)
                        test(n, m, alpha, seed)
    bin_df = pd.DataFrame.from_dict(results_bin)
    bin_df = bin_df.groupby(['n','m','alpha'], as_index=False).mean()
    bin_df = bin_df.drop(columns = ['seed'])
    bin_df['time'] = bin_df['time'].round(5)
    bin_df['num_nodes'] = bin_df['num_nodes'].round(2)
    bin_df['num_candidates'] = bin_df['num_candidates'].round(2)
    bin_df.to_csv('bin_data.txt', sep='\t', index=False)

    num_df = pd.DataFrame.from_dict(results_num)
    num_df = num_df.groupby(['n','m','alpha'], as_index=False).mean()
    num_df = num_df.drop(columns = ['seed'])
    num_df['time'] = num_df['time'].round(5)
    num_df['num_nodes'] = num_df['num_nodes'].round(2)
    num_df['num_candidates'] = num_df['num_candidates'].round(2)
    num_df.to_csv('num_data.txt', sep='\t', index=False)
    
    
    #bin_df = pd.read_csv('bin_data.txt', sep='\t')
    #num_df = pd.read_csv('num_data.txt', sep='\t')
    plot_results(bin_df, num_df, 'n','time', {'m','alpha'})
    plot_results(bin_df, num_df,'m','time', {'n','alpha'})
    plot_results(bin_df, num_df,'alpha','time', {'n','m'})
    plot_results(bin_df, num_df,'n','num_nodes', {'m','alpha'})
    plot_results(bin_df, num_df,'m','num_nodes', {'n','alpha'})
    plot_results(bin_df, num_df,'alpha','num_nodes', {'n','m'})
    plot_results(bin_df, num_df,'n','num_candidates', {'m','alpha'})
    plot_results(bin_df, num_df,'m','num_candidates', {'n','alpha'})
    plot_results(bin_df, num_df,'alpha','num_candidates', {'n','m'})
    plt.show()