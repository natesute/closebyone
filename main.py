import timeit
import numpy as np
from numba import int64, float64
from numba.core.types import Array
# from numba import njit
# from numba.experimental import jitclass

RNG = np.random.default_rng(seed=0)

# convert discrete numerical dataset to binary
def disc_num_to_bin(num_d_ref):
    num_d = np.copy(num_d_ref)
    bin_d = []
    for att in range(len(num_d[0])-1, -1, -1):
        col = num_d[:, att]
        thresholds = np.unique(col)
        for i in range(len(thresholds)-1, -1, -1):
            new_col = np.copy(col)
            new_col[col <= thresholds[i]] = True
            new_col[col > thresholds[i]] = False
            bin_d.append(new_col)
        for i in range(0, len(thresholds)):
            new_col = np.copy(col)
            new_col[col >= thresholds[i]] = True
            new_col[col < thresholds[i]] = False
            bin_d.append(new_col)
    bin_d = np.array(bin_d, dtype=bool).T
    return bin_d

# convert continuous numerical dataset to binary dataset
def cont_num_to_bin(num_d_ref):
    num_d = np.copy(num_d_ref)
    bin_d = []
    for att in range(len(num_d[0])):
        col = num_d[:, att]
        sorted_col = np.sort(col)
        for val in sorted_col:
            new_col = np.copy(col)
            new_col[col <= val] = True
            new_col[col > val] = False
            bin_d.append(new_col)
        for val in sorted_col:
            new_col = np.copy(col)
            new_col[col < val] = False
            new_col[col >= val] = True
            bin_d.append(new_col)
    bin_d = np.array(bin_d, dtype=bool).T
    return bin_d

# create random discrete numerical array
def rand_disc_num_array(rows, cols, alpha_=0.5):
    target_col = np.zeros((rows, 1))
    end_of_1s = int(len(target_col) * alpha_)
    for i in range(end_of_1s):
        target_col[i] = 1
    RNG.shuffle(target_col)
    data_cols = np.random.randint(1, 10, (rows, cols))
    return data_cols, target_col

# create random continuous numerical array
def rand_cont_num_array(rows, cols, alpha_=0.5):
    target_col = np.zeros((cols, 1))
    end_of_1s = int(len(target_col) * alpha_)
    for i in range(end_of_1s):
        target_col[i] = 1
    RNG.shuffle(target_col)
    data_cols = np.random.rand(rows, cols)
    return data_cols, target_col


class BB:
    num_nodes: int64
    num_candidates: int64
    max_obj: float64
    p0: float64
    bnd0: float64
    n: int64
    data: Array
    target: Array
    best_query: Array

    def __init__(self, num_nodes, num_candidates, max_obj, p0, bnd0, n_data, data, target, best_query):
        self.num_nodes = num_nodes
        self.num_candidates = num_candidates
        self.max_obj = max_obj
        self.p0 = p0
        self.bnd0 = bnd0
        self.n = n_data
        self.data = data
        self.target = target
        self.best_query = best_query

    def __repr__(self):
        return f'{self.num_patterns} patterns\n{self.num_nodes} nodes\n{self.num_candidates} candidates'

"""
def get_root(d):
    rt = np.empty((d, 2))
    for i in range(d):
        rt[i] = np.array([0, 9])
    return rt
"""

def p(extent, bb):
    target_col = bb.target[extent]
    return target_col.mean()


def impact(extent, bb):
    return (len(extent) / bb.n) * (p(extent, bb) - bb.p0)


def bnd(extent, bb):
    target_col = bb.target[extent]
    return (target_col.sum() / bb.n) * (1 - bb.p0)


def get_mins(extent, bb):
    """ IF USING NO-PYTHON NUMBA
    dims = len(data[0])
    mins = np.empty(dims)
    for j in range(dims):
        _min = np.inf
        for datum in data:
            temp = datum[j]
            if temp < _min:
                _min = temp
        mins[j] = _min
    return mins
    """
    data_in_extent = bb.data[extent]
    return np.min(data_in_extent, axis=0)


def get_maxs(extent, bb):
    """ IF USING NO-PYTHON NUMBA
    dims = len(data[0])
    maxs = np.empty(dims)
    for j in range(dims):
        _max = np.NINF
        for datum in data:
            temp = datum[j]
            if temp > _max:
                _max = temp
        maxs[j] = _max
    return maxs
    """
    data_in_extent = bb.data[extent]
    return np.max(data_in_extent, axis=0)


def get_extent(intent, extent, bb):
    dims = len(bb.data[0])
    for j in range(dims):
        low = intent[j][0]
        high = intent[j][1]
        mask = (bb.data[extent, j] >= low) & (bb.data[extent, j] <= high)
        extent = extent[mask]
    return extent


def get_closure(extent, bb):
    dims = len(bb.data[0])
    new_intent = np.empty((dims, 2))
    mins = get_mins(extent, bb)
    maxs = get_maxs(extent, bb)

    for j in range(dims):
        new_intent[j][0] = mins[j]
        new_intent[j][1] = maxs[j]

    return new_intent


def is_canonical(intent, new_intent, current_j):
    for j in range(current_j + 1, len(intent[0])):
        # if a bound has been changed in a previous (greater than current j) attribute
        if intent[j][0] != new_intent[j][0] or intent[j][1] != new_intent[j][1]:
            return False
    return True


def search_plus_lower(intent, extent, j, bb):
#     print("plus lower\n")
    # raise lower bound of j
    intent[j][0] += 1
    extent = get_extent(intent, extent, bb)

    if extent.size > 0:
        bb.max_obj = max(impact(extent, bb), bb.max_obj)
#         print("intent")
#         print(intent)
#         print("max obj")
#         print(bb.max_obj)
#         print("\n\n")
        if bnd(extent, bb) >= bb.max_obj: # consider >. > would mean that, if peaking, peaked nodes would not be included at all.
#             print("bound accepted\n")
            # get closure, returns empty if not canonical
            new_intent = get_closure(extent, bb)
            # check if new intent is canonical

            if is_canonical(intent, new_intent, j):
#                 print("is canonical\n")
                #print("before closure")
                #print(intent)
                intent = new_intent
                print("closure")
                print(intent)
                bb.num_nodes += 1
                
                # print(intent)
                # check if bounds can be further changed on j
                if intent[j][0] != intent[j][1]:
                    # branch current attribute, both bounds
                    search_minus_upper(np.copy(intent), np.copy(extent), j, bb)
                    # check if maximum bound has been hit
                    if bb.max_obj == bb.bnd0:
                        return
                    search_plus_lower(np.copy(intent), np.copy(extent), j, bb)

                    if bb.max_obj == bb.bnd0:
                        return
                if j:
                    search(np.copy(intent), np.copy(extent), j - 1, bb)


def search_minus_upper(intent, extent, j, bb):
#     print("minus upper\n")
    # lower upper bound of j
    intent[j][1] -= 1
    extent = get_extent(np.copy(intent), np.copy(extent), bb)
    if extent.size > 0:
        bb.max_obj = max(impact(extent, bb), bb.max_obj)
#         print("intent")
#         print(intent)
#         print("max obj")
#         print(bb.max_obj)
#         print("\n\n")
        #if bb.max_obj > old_max_obj:
        #    best_query = intent
        if bnd(extent, bb) >= bb.max_obj:
#             print("bound accepted\n")
            # get closure, returns empty if not canonical
            new_intent = get_closure(extent, bb)

            if is_canonical(intent, new_intent, j):
#                 print("is canonical\n")
                #print("before closure")
                #print(intent)
                intent = new_intent
                print("closure")
                print(intent)
                
                bb.num_nodes += 1
                # check if bounds can be further changed on j
                if intent[j][0] != intent[j][1]:
                    # branch current attribute, only upper bound
                    search_minus_upper(np.copy(intent), np.copy(extent), j, bb)
                    # check if maximum bound has been hit
                    if bb.max_obj == bb.bnd0:
                        return

                if j:  # because j==0 is value col
                    search(np.copy(intent), np.copy(extent), j - 1, bb)

# search changes on jth attribute downwards
def search(intent, extent, j, bb):
#     print("next attribute down\n")
    if intent[j][0] != intent[j][1]:
        # search aug children of minus-ing upper bound
        search_minus_upper(np.copy(intent), np.copy(extent), j, bb)
        # check if maximum bound has been hit
        if bb.max_obj == bb.bnd0:
            return

        # search aug children of plus-ing lower bound
        search_plus_lower(np.copy(intent), np.copy(extent), j, bb)
        # check if maximum bound has been hit
        if bb.max_obj == bb.bnd0:
            return

    if j > 0:
        # search changes on j-1th attribute downwards
        search(np.copy(intent), np.copy(extent), j - 1, bb)

def run_cbo(data, target):
    extent = np.arange(len(data))
    bb = BB(0, 0, np.NINF, 0, 0, len(data), data, target, None)
    bb.p0 = p(extent, bb)
    bb.bnd0 = bnd(extent, bb)
    dims = len(data[0])
    # print(bb)
    # intent = get_root(dims)
    print("start")
    print(extent)
    print("data")
    print(bb.data)
    intent = get_closure(extent, bb)
    extent = get_extent(np.copy(intent), np.copy(extent), bb)
    bb.max_obj = max(impact(extent, bb), bb.max_obj)
    print("closure")
    print(intent)
#     print("max obj")
#     print(bb.max_obj)
#     print("\n\n")

    # check if inital objective value is maximum bound
    if bb.max_obj == bb.bnd0:
        return bb

    # intent = get_closure(extent, bb)
    bb.num_nodes += 1
    # print(intent)

    # get index of last attribute

    # starts from last attribute (index dims-1) and decrements
    j = dims-1
    search(np.copy(intent), np.copy(extent), j, bb)

    return bb


def search_plus_lower_nc(intent, extent, j, bb):
#     print("plus lower\n")
    # raise lower bound of j
    intent[j][0] += 1
    extent = get_extent(intent, extent, bb)

    if extent.size > 0:
        bb.max_obj = max(impact(extent, bb), bb.max_obj)
#         print("intent")
#         print(intent)
#         print("max obj")
#         print(bb.max_obj)
#         print("\n\n")
        if bnd(extent, bb) > bb.max_obj: # consider >. > would mean that, if peaking, peaked nodes would not be included at all.
#             print("bound accepted\n")
#             print("closure")
#             print(intent)
            # check if new intent is canonical
            intent = get_closure(extent, bb)
            bb.num_nodes += 1
            
            # print(intent)
            # check if bounds can be further changed on j
            if intent[j][0] != intent[j][1]:
                # branch current attribute, both bounds
                search_minus_upper_nc(np.copy(intent), np.copy(extent), j, bb)
                # check if maximum bound has been hit
                if bb.max_obj == bb.bnd0:
                    return
                search_plus_lower_nc(np.copy(intent), np.copy(extent), j, bb)

                if bb.max_obj == bb.bnd0:
                    return
            if j:
                search_nc(np.copy(intent), np.copy(extent), j - 1, bb)


def search_minus_upper_nc(intent, extent, j, bb):
#     print("minus upper\n")
    # lower upper bound of j
    intent[j][1] -= 1
    extent = get_extent(np.copy(intent), np.copy(extent), bb)
    if extent.size > 0:
        bb.max_obj = max(impact(extent, bb), bb.max_obj)
#         print("intent")
#         print(intent)
#         print("max obj")
#         print(bb.max_obj)
#         print("\n\n")
        #if bb.max_obj > old_max_obj:
        #    best_query = intent
        if bnd(extent, bb) > bb.max_obj:
#             print("bound accepted\n")
            # get closure, returns empty if not canonical
            intent = get_closure(extent, bb)
#             print("closure")
#             print(intent)
            
            bb.num_nodes += 1
            # check if bounds can be further changed on j
            if intent[j][0] != intent[j][1]:
                # branch current attribute, only upper bound
                search_minus_upper_nc(np.copy(intent), np.copy(extent), j, bb)
                # check if maximum bound has been hit
                if bb.max_obj == bb.bnd0:
                    return

            if j:  # because j==0 is value col
                search(np.copy(intent), np.copy(extent), j - 1, bb)

# search changes on jth attribute downwards
def search_nc(intent, extent, j, bb):
#     print("next attribute down\n")
    if intent[j][0] != intent[j][1]:
        # search aug children of minus-ing upper bound
        search_minus_upper_nc(np.copy(intent), np.copy(extent), j, bb)
        # check if maximum bound has been hit
        if bb.max_obj == bb.bnd0:
            return

        # search aug children of plus-ing lower bound
        search_plus_lower_nc(np.copy(intent), np.copy(extent), j, bb)
        # check if maximum bound has been hit
        if bb.max_obj == bb.bnd0:
            return

    if j > 0:
        # search changes on j-1th attribute downwards
        search_nc(np.copy(intent), np.copy(extent), j - 1, bb)


def run_cbo_nc(data, target):
    extent = np.arange(len(data))
    bb = BB(0, 0, np.NINF, 0, 0, len(data), data, target, None)
    bb.p0 = p(extent, bb)
    bb.bnd0 = bnd(extent, bb)
    dims = len(data[0])
    # print(bb)
    # intent = get_root(dims)
    intent = get_closure(extent, bb)
    extent = get_extent(np.copy(intent), np.copy(extent), bb)
    bb.max_obj = max(impact(extent, bb), bb.max_obj)
#     print("intent")
#     print(intent)
#     print("max obj")
#     print(bb.max_obj)
#     print("\n\n")

    # check if inital objective value is maximum bound
    if bb.max_obj == bb.bnd0:
        return bb

    # intent = get_closure(extent, bb)
    bb.num_nodes += 1
    # print(intent)

    # get index of last attribute

    # starts from last attribute (index dims-1) and decrements
    j = dims-1

    search_nc(np.copy(intent), np.copy(extent), j, bb)

    return bb


# baseline approach


def root(data):
    return np.add.reduce(data, axis=0) == len(data)


def implied_on(j, ext, bb):
    for i in ext:
        if not bb.data[i, j]:
            return False
    return True


def baseline(target, ext_sizes, query, extent, i, bb):
    print("query")
    print(query)
    print("extent")
    print(extent)
    print("max obj")
    print(bb.max_obj)
    bb.num_nodes += 1
    num = len(query)

    for prop in range(i, num):
        if bb.max_obj == bb.bnd0:
            return
        aug_extent = extent[bb.data[extent, prop]]
        if aug_extent.size == 0:
            continue
        # proposition is already implied
        if query[prop] == True:
            continue
        bb.num_candidates += 1
        aug_query = query.copy()
        aug_query[prop] = True
        print("tail augmentation")
        print(aug_query)

        bb.max_obj = max(impact(aug_extent, bb), bb.max_obj)

        if bnd(aug_extent, bb) < bb.max_obj:
            continue

        prefix_pres = True
        for k in range(prop):
            if len(aug_extent) <= ext_sizes[k] and implied_on(k, aug_extent, bb):
                prefix_pres = False
                break
        if not prefix_pres:
            continue

        for k in range(prop + 1, num):
            if len(aug_extent) <= ext_sizes[k] and implied_on(k, aug_extent, bb):
                aug_query[k] = True
        print("HELLO!")
        print("aug query")
        print(aug_query)
        baseline(target, ext_sizes, aug_query, aug_extent, prop + 1, bb)


def run_baseline(data, target):
    root_query = root(data)
    extent = np.arange(len(data))
    ext_sizes = np.add.reduce(data)
    bb = BB(0, 0, np.NINF, 0, 0, len(data), data, target, None)
    bb.p0 = p(extent, bb)
    bb.bnd0 = bnd(extent, bb)
    baseline(target, ext_sizes, root_query, extent, 0, bb)
    return bb


def run_cbo_random(num_ds, targets):
    n = np.random.randint(1000)
    data = num_ds[n]
    target = targets[n]
    return run_cbo(data, target)

def run_cbo_nc_random(num_ds, targets):
    n = np.random.randint(1000)
    data = num_ds[n]
    target = targets[n]
    return run_cbo_nc(data, target)

def run_baseline_random(bin_ds, targets):
    n = np.random.randint(1000)
    data = bin_ds[n]
    target = targets[n]
    return run_baseline(data, target)
    
def random_results(i, j, alpha):
    num_d = [None]*1000
    bin_d = [None]*1000
    targets = [None]*1000
    for n in range(1000):
        num_d[n], targets[n] = rand_disc_num_array(i, j, alpha)
        bin_d[n] = disc_num_to_bin(num_d[n]).copy()
    t_bin = timeit.timeit('run_baseline_random(bin_d, targets)')
    t_num = timeit.timeit('run_cbo_random(num_d, targets)')
    return t_num, t_bin


# driver code
if __name__ == '__main__':
    """
    no_of_attr = np.arange(2, 20, 2)
    # m = np.arange(20, 81, 20)
    no_of_data = np.arange(2, 20, 2)
    # alpha = np.arange(0.25, 0.5, 0.75)
    alpha = 0.8

    num_data = {}
    bin_data = {}
    targets = {}
    for m in no_of_data:
        for n in no_of_attr:
            # get numerical data and target columns for different data num and attr num
            num_data[m, n], targets[m, n] = rand_disc_num_array(m, n, alpha)
            # convert and store binary data from num
            bin_data[m, n] = cont_num_to_bin(num_data[m, n]).copy()

    m = 4
    n = 4
    
    
    result = run_baseline(bin_data[m, n], targets[m, n])
    print("target")
    print(targets[m,n])
    print("\n")
    print(f"data_bin[{m},{n}]")
    print(bin_data[m, n])
    print("max obj")
    print(result.max_obj)

    result = run_cbo(num_data[m, n], targets[m, n])
    print(f"data_num[{m},{n}]")
    print(num_data[m, n])
    print("max obj")
    print(result.max_obj)
    """
    """
    data = np.array([[7,7],[9,4]])
    target = np.array([1,0])
    result = cbo(data,target)
    print(result.num_patterns)
    """
    
    num_d = np.array([[1,3],[2,2],[3,1]])
    target = np.array([1,1,0])
    print("target")
    print(target)
    #print("numd", num_d)
    # result = run_cbo(num_d, target)
    # print("max obj")
    # print(result.max_obj)

    bin_d = disc_num_to_bin(num_d)
    print("bind", bin_d)
    result = run_baseline(bin_d, target)
    print("max obj")
    print(result.max_obj)