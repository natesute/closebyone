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
    # range through attributes (reversed)
    for att in range(len(num_d[0])-1, -1, -1):
        col = num_d[:, att]
        thresholds = np.unique(col)
        # range through ordered threshold values (reversed)
        for i in range(len(thresholds)-1, -1, -1):
            new_col = np.copy(col)
            # binarise column values according to proposition (<=)
            new_col[col <= thresholds[i]] = True
            new_col[col > thresholds[i]] = False
            bin_d.append(new_col)
        # range through ordered threshold values
        for i in range(len(thresholds)):
            new_col = np.copy(col)
            # binarise column values according to proposition (>=)
            new_col[col >= thresholds[i]] = True
            new_col[col < thresholds[i]] = False
            bin_d.append(new_col)
    # columns were appended to bin_d, so must be transposed to actually be columns
    bin_d = np.array(bin_d, dtype=bool).T
    return bin_d

# convert continuous numerical dataset to binary dataset
def cont_num_to_bin(num_d_ref):
    num_d = np.copy(num_d_ref)
    bin_d = []
    for att in range(0, len(num_d[0])-1, -1):
        col = num_d[:, att]
        thresholds = np.sort(col)
        for i in thresholds:
            new_col = np.copy(col)
            new_col[col <= thresholds[i]] = True
            new_col[col > thresholds[i]] = False
            bin_d.append(new_col)
        for i in thresholds:
            new_col = np.copy(col)
            new_col[col >= thresholds[i]] = True
            new_col[col < thresholds[i]] = False
            bin_d.append(new_col)
    bin_d = np.array(bin_d, dtype=bool).T
    return bin_d

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

# create random continuous numerical array
def rand_cont_num_array(rows, cols):
    data_cols = np.random.rand(rows, cols)
    return data_cols


class BB:
    num_nodes: int64
    num_edges: int64
    max_obj: float64
    p0: float64
    bnd0: float64
    n_data: int64
    n_att: int64
    data: Array
    target: Array
    best_query: Array
    time: float64

    def __init__(self, num_nodes, num_edges, max_obj, p0, bnd0, n_data, n_att, data, target, best_query, time):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.max_obj = max_obj
        self.p0 = p0
        self.bnd0 = bnd0
        self.n_data = n_data
        self.n_att = n_att
        self.data = data
        self.target = target
        self.best_query = best_query
        self.time = time

    def __repr__(self):
        return f'{self.time} time\n{self.num_nodes} nodes\n{self.num_edges} edges'

def p(extent, bb):
    target_col = bb.target[extent]
    return target_col.mean()


def impact(extent, bb):
    return (len(extent) / bb.n_data) * (p(extent, bb) - bb.p0)


def bnd(extent, bb):
    target_col = bb.target[extent]
    return (target_col.sum() / bb.n_data) * (1 - bb.p0)


def get_mins(extent, bb):
    data_in_extent = bb.data[extent]
    return np.min(data_in_extent, axis=0)


def get_maxs(extent, bb):
    data_in_extent = bb.data[extent]
    return np.max(data_in_extent, axis=0)


def get_extent(intent, extent, bb):
    for j in range(bb.n_att):
        low = intent[j][0]
        high = intent[j][1]
        mask = (bb.data[extent, j] >= low) & (bb.data[extent, j] <= high)
        extent = extent[mask]
    return extent


def get_closure(extent, bb):
    new_intent = np.empty((bb.n_att, 2))
    mins = get_mins(extent, bb)
    maxs = get_maxs(extent, bb)

    for j in range(bb.n_att):
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
    # raise lower bound of j
    bb.num_edges += 1
    intent[j][0] += 1
    extent = get_extent(intent, extent, bb)

    if extent.size > 0:
        bb.max_obj = max(impact(extent, bb), bb.max_obj)
        if bb.max_obj == bb.bnd0:
            bb.num_nodes += 1
            return

        if bnd(extent, bb) >= bb.max_obj:
            new_intent = get_closure(extent, bb)
            
            # check if new intent is canonical
            if is_canonical(intent, new_intent, j):
                intent = new_intent
                bb.num_nodes += 1
                
                # print(intent)
                # check if bounds can be further changed on j
                if intent[j][0] != intent[j][1]:
                    # branch current attribute, both bounds
                    search_minus_upper(np.copy(intent), np.copy(extent), j, bb)
                    
                    search_plus_lower(np.copy(intent), np.copy(extent), j, bb)

                if j:
                    search(np.copy(intent), np.copy(extent), j - 1, bb)


def search_minus_upper(intent, extent, j, bb):
    # lower upper bound of j
    bb.num_edges += 1
    intent[j][1] -= 1
    extent = get_extent(np.copy(intent), np.copy(extent), bb)
    if extent.size > 0:
        bb.max_obj = max(impact(extent, bb), bb.max_obj)
        # check if maximum bound has been hit
        if bb.max_obj == bb.bnd0:
            bb.num_nodes += 1
            return
        if bnd(extent, bb) >= bb.max_obj:
            new_intent = get_closure(extent, bb)

            if is_canonical(intent, new_intent, j):
#                 print("is canonical\n")
                #print("before closure")
                #print(intent)
                intent = new_intent
#                 print("closure")
#                 print(intent)
                
                bb.num_nodes += 1
                # check if bounds can be further changed on j
                if intent[j][0] != intent[j][1]:
                    # branch current attribute, only upper bound
                    search_minus_upper(np.copy(intent), np.copy(extent), j, bb)
                    

                if j:  # because j==0 is value col
                    search(np.copy(intent), np.copy(extent), j - 1, bb)

# search changes on jth attribute downwards
def search(intent, extent, j, bb):
#     print("next attribute down\n")
    if intent[j][0] != intent[j][1]:
        # search aug children of minus-ing upper bound
        search_minus_upper(np.copy(intent), np.copy(extent), j, bb)

        # search aug children of plus-ing lower bound
        search_plus_lower(np.copy(intent), np.copy(extent), j, bb)

    if j > 0:
        # search changes on j-1th attribute downwards
        search(np.copy(intent), np.copy(extent), j - 1, bb)

def run_cbo(data, target):
    extent = np.arange(len(data))
    bb = BB(0, 0, np.NINF, 0, 0, len(data), len(data[0]), data, target, None, 0)
    bb.p0 = p(extent, bb)
    bb.bnd0 = bnd(extent, bb)
    intent = get_closure(extent, bb)
    bb.max_obj = max(impact(extent, bb), bb.max_obj)
    
    bb.num_nodes += 1
    
    # check if inital objective value is maximum bound
    if bb.max_obj == bb.bnd0:
        return bb

    # starts searching from last attribute and decrements
    j = bb.n_att-1
    def closure():
        return search(np.copy(intent), np.copy(extent), j, bb)
    bb.time = timeit.timeit(closure, number=1)
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
    bb.num_nodes += 1
    num = len(query)

    for prop in range(i, num):
        bb.num_edges += 1
        if bb.max_obj == bb.bnd0:
            return
        aug_extent = extent[bb.data[extent, prop]]
        if aug_extent.size == 0:
            continue
        # proposition is already implied
        if query[prop] == True:
            continue
        bb.num_edges += 1
        aug_query = query.copy()
        aug_query[prop] = True
#         print("tail augmentation")
#         print(aug_query)

        bb.max_obj = max(impact(aug_extent, bb), bb.max_obj)

        if bb.max_obj > bnd(aug_extent, bb):
            continue

        prefix_pres = True
        for k in range(prop):
            if query[k]:
                pass
            elif len(aug_extent) <= ext_sizes[k] and implied_on(k, aug_extent, bb):
                prefix_pres = False
                break
        if not prefix_pres:
            continue

        for k in range(prop + 1, num):
            if len(aug_extent) <= ext_sizes[k] and implied_on(k, aug_extent, bb):
                aug_query[k] = True
        baseline(target, ext_sizes, aug_query, aug_extent, prop + 1, bb)


def run_baseline(data, target):
    root_query = root(data)
    extent = np.arange(len(data))
    ext_sizes = np.add.reduce(data)
    bb = BB(0, 0, np.NINF, 0, 0, len(data), len(data[0]), data, target, None, 0)
    bb.p0 = p(extent, bb)
    bb.bnd0 = bnd(extent, bb)

    def closure():
        return baseline(target, ext_sizes, root_query, extent, 0, bb)
    bb.time = timeit.timeit(closure, number=1)

    return bb