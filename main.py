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
    for att in range(len(num_d[0])):
        col = num_d[:, att]
        unique = np.unique(col)
        for val in unique:
            new_col = np.copy(col)
            new_col[col <= val] = 1
            new_col[col > val] = 0
            bin_d.append(new_col)
        for val in unique:
            new_col = np.copy(col)
            new_col[col < val] = 0
            new_col[col >= val] = 1
            bin_d.append(new_col)
    bin_d = np.array(bin_d).T
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
            new_col[col <= val] = 1
            new_col[col > val] = 0
            bin_d.append(new_col)
        for val in sorted_col:
            new_col = np.copy(col)
            new_col[col < val] = 0
            new_col[col >= val] = 1
            bin_d.append(new_col)
    bin_d = np.array(bin_d).T
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
    num_patterns: int64
    max_obj: float64
    p0: float64
    n: int64
    data: Array
    target: Array
    best_query: Array

    def __init__(self, num_nodes, num_candidates, num_patterns, max_obj, p0, n_data, data, target, best_query):
        self.num_nodes = num_nodes
        self.num_candidates = num_candidates
        self.num_patterns = num_patterns
        self.max_obj = max_obj
        self.p0 = p0
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


def plus_lower(intent, extent, j, bb):
    # raise lower bound of j
    intent[j][0] += 1
    extent = get_extent(intent, extent, bb)

    if extent.size > 0:
        bb.max_obj = max(impact(extent, bb), bb.max_obj)
        if bnd(extent, bb) >= bb.max_obj:
            # get closure, returns empty if not canonical
            new_intent = get_closure(extent, bb)
            # check if new intent is canonical

            if is_canonical(intent, new_intent, j):
                intent = new_intent
                # print("intent")
                # print(intent)
                # print("max obj")
                # print(bb.max_obj)
                # print("\n\n")
                bb.num_patterns += 1
                # print(intent)
                # check if bounds can be further changed on j
                if intent[j][0] != intent[j][1]:
                    # branch current attribute, both bounds
                    minus_upper(np.copy(intent), np.copy(extent), j, bb)
                    plus_lower(np.copy(intent), np.copy(extent), j, bb)
                if j:
                    previous_js(np.copy(intent), np.copy(extent), j - 1, bb)


def minus_upper(intent, extent, j, bb):
    # lower upper bound of j
    intent[j][1] -= 1
    extent = get_extent(np.copy(intent), np.copy(extent), bb)
    if extent.size > 0:
        old_max_obj = bb.max_obj
        bb.max_obj = max(impact(extent, bb), old_max_obj)
        #if bb.max_obj > old_max_obj:
        #    best_query = intent
        if bnd(extent, bb) >= bb.max_obj:
            # get closure, returns empty if not canonical
            new_intent = get_closure(extent, bb)

            if is_canonical(intent, new_intent, j):
                intent = new_intent
                """
                print("intent")
                print(intent)
                print("max obj")
                print(bb.max_obj)
                print("\n\n")
                """
                bb.num_patterns += 1
                # check if bounds can be further changed on j
                if intent[j][0] != intent[j][1]:
                    # branch current attribute, only upper bound
                    minus_upper(np.copy(intent), np.copy(extent), j, bb)
                if j == 1:  # because j==0 is value col
                    previous_js(np.copy(intent), np.copy(extent), j - 1, bb)


def previous_js(intent, extent, j, bb):
    if intent[j][0] != intent[j][1]:
        minus_upper(np.copy(intent), np.copy(extent), j, bb)
        plus_lower(np.copy(intent), np.copy(extent), j, bb)
    if j > 0:
        previous_js(np.copy(intent), np.copy(extent), j - 1, bb)


def search(intent, extent, j, bb):
    minus_upper(np.copy(intent), np.copy(extent), j, bb)
    plus_lower(np.copy(intent), np.copy(extent), j, bb)
    if j > 0:
        search(np.copy(intent), np.copy(extent), j - 1, bb)


def cbo(data, target):
    extent = np.arange(len(data))
    bb = BB(0, 0, 0, np.NINF, 0, len(data), data, target, None)
    bb.p0 = p(extent, bb)
    dims = len(data[0])
    # print(bb)
    # intent = get_root(dims)
    intent = get_closure(extent, bb)
    # print("intent")
    # print(intent)
    # print("max obj")
    # print(bb.max_obj)
    # print("\n\n")
    extent = get_extent(np.copy(intent), np.copy(extent), bb)
    bb.max_obj = max(impact(extent, bb), bb.max_obj)
    if bnd(extent, bb) <= bb.max_obj:
        return bb
    # get closure, returns empty if not canonical
    # intent = get_closure(extent, bb)
    bb.num_patterns += 1
    # print(intent)

    # get index of last attribute

    # starts from last attribute (index dims-1) and decrements
    j = dims-1

    # search(np.copy(intent), np.copy(extent), j, bb)
    minus_upper(np.copy(intent), np.copy(extent), j, bb)
    plus_lower(np.copy(intent), np.copy(extent), j, bb)
    previous_js(np.copy(intent), np.copy(extent), j-1, bb)
    return bb


# baseline approach


def root(data):
    return np.add.reduce(data, axis=0) == len(data)


def implied_on(j, ext, bb):
    for i in ext:
        if not bb.data[i, j]:
            return False
    return True


def cbo_generate(target, s, r, extent, i, bb):
    bb.num_nodes += 1
    num = len(r)

    for j in range(i, num):
        if r[j]:
            continue
        bb.num_candidates += 1
        _r = r.copy()
        _r[j] = True
        aug_extent = extent[bb.data[extent, j]]
        bb.max_obj = max(impact(aug_extent, bb), bb.max_obj)

        if bnd(aug_extent, bb) < bb.max_obj:
            continue

        pp = True
        for k in range(j):
            if r[k]:
                _r[k] = True
            elif len(aug_extent) <= s[k] and implied_on(k, aug_extent, bb):
                pp = False
                break
        if not pp:
            continue

        for k in range(j + 1, num):
            if len(aug_extent) <= s[k] and implied_on(k, aug_extent, bb):
                _r[k] = True

        cbo_generate(target, s, _r, aug_extent, j + 1, bb)


def cbo_baseline(data, target):
    rt = root(data)
    extent = np.arange(len(data))
    ext_sizes = np.add.reduce(data)
    bb = BB(0, 0, 0, np.NINF, 0, len(data), data, target, None)
    bb.p0 = p(extent, bb)
    cbo_generate(target, ext_sizes, rt, extent, 0, bb)
    return bb


# driver code
if __name__ == '__main__':
    no_of_attr = np.arange(2, 9, 2)
    # m = np.arange(20, 81, 20)
    no_of_data = np.arange(2, 13, 2)
    # alpha = np.arange(0.25, 0.5, 0.75)
    alpha = 0.5

    num_data = {}
    bin_data = {}
    targets = {}
    for m in no_of_data:
        for n in no_of_attr:
            # get numerical data and target columns for different data num and attr num
            num_data[m, n], targets[m, n] = rand_disc_num_array(m, n, alpha)
            # convert and store binary data from num
            bin_data[m, n] = cont_num_to_bin(num_data[m, n]).copy()

    m = 2
    n = 2
    """
    result = cbo_baseline(bin_data[m, n], targets[m, n])
    print("target")
    print(targets[m,n])
    print("\n")
    print(f"data_bin[{m},{n}]")
    print(bin_data[m, n])
    print("max obj")
    print(result.max_obj)

    result = cbo(num_data[m, n], targets[m, n])
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
    num_d = np.array([[1,1],[1,2]])
    print("numd", num_d)
    bin_d = disc_num_to_bin(num_d)
    print("numd", num_d)
    target = np.array([1,0])
    result = cbo_baseline(bin_d, target)
    print("target")
    print(targets[m,n])
    print("\n")
    print(f"data_bin")
    print(bin_d)
    print("max obj")
    print(result.max_obj)

    result = cbo(num_d, target)
    print(f"data_num")
    print(num_d)
    print("max obj")
    print(result.max_obj)