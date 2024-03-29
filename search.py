import numpy as np # type: ignore
import copy
from realkd.logic import TabulatedProposition
import sortednp as snp
from numpy import array
from bitarray import bitarray
from bitarray.util import subset
from collections import defaultdict
import pandas as pd
from math import inf


class PropSearch:
    def __init__(self, curr_node, context, res):
        self.curr_node = curr_node
        self.context = context
        self.res = res


class Results:
    def __init__(self):
        self.num_candidates = 0
        self.num_nodes = 0
        self.max_obj = -inf
        self.opt_node = None
        self.time = 0

    def __repr__(self):
        repr_str = ""
        repr_str += "Num Candidates: " + str(self.num_candidates) + "\n"
        repr_str += "Num Nodes: " + str(self.num_nodes) + "\n"
        repr_str += "Max Obj: " + str(self.max_obj) + "\n"
        repr_str += "\n**********Best Node**********\n" + str(self.best_node) + "*****************************\n\n"
        repr_str += "Time: " + str(self.time) + "\n"
        return repr_str


class Intent:
    def __init__(self, pattern=None):
        self.pattern = pattern
    
    def __repr__(self):
        repr = []
        for attr in self.pattern:
            repr.append(f"[{attr[0]:.0f}, {attr[1]:.0f}]")

        return f"<{(', ').join(repr)}>"

    def __len__(self):
        return len(self.pattern)

    def __getitem__(self, index):
        return self.pattern[index]
    
    def fully_closed(self, j):
        return self.pattern[j][0] == self.pattern[j][1]


class Query:
    def __init__(self, props):
        self.props = props

    def __len__(self):
        return len(self.props)

    def __getitem__(self, index):
        return self.props[index]
    
    def __setitem__(self, index, value):
        self.props[index] = value
    
    def __str__(self):
        str = []
        for prop in self.props:
            str.append(f"{prop:.0f}")

        return f"<{(', ').join(str)}>"


class Utilities:

    @staticmethod
    def is_canonical(curr_intent: Intent, new_intent: Intent, j):
        for i in range(j + 1, len(curr_intent)):
            # if a bound has been changed in a previous (greater than current j) attribute
            if curr_intent[i][0] != new_intent[i][0] or curr_intent[i][1] != new_intent[i][1]:
                return False
        return True

    @staticmethod
    def disc_num_to_bin(old_objects_num):
        '''converts discrete numerical dataset to binary'''
        objects_num = np.copy(old_objects_num)
        objects_bin = []
        props = []
        # range through attributes (reversed)
        for att in range(len(objects_num[0])-1, -1, -1):
            col = objects_num[:, att]
            thresholds = np.unique(col)
            # range through ordered threshold values (reversed)
            for i in range(len(thresholds)-1, -1, -1):
                new_col = np.copy(col)
                # binarise column values according to proposition (<=)
                new_col[col <= thresholds[i]] = True
                new_col[col > thresholds[i]] = False
                objects_bin.append(new_col)
                props.append(f"x({att}) <= {thresholds[i]}")
            # range through ordered threshold values
            for i in range(len(thresholds)):
                new_col = np.copy(col)
                # binarise column values according to proposition (>=)
                new_col[col >= thresholds[i]] = True
                new_col[col < thresholds[i]] = False
                objects_bin.append(new_col)
                props.append(f"x({att}) >= {thresholds[i]}")
        # columns were appended to bin_d, so must be transposed to actually be columns
        objects_bin = np.array(objects_bin, dtype=bool).T
        # props = np.array(props, dtype=str)
        return objects_bin

    @staticmethod
    def cont_num_to_bin(old_objects_num):
        '''converts continuous numerical dataset to binary'''
        objects_num = np.copy(old_objects_num)
        objects_bin = []
        for att in range(0, len(objects_num[0])-1, -1):
            col = objects_num[:, att]
            thresholds = np.sort(col)
            for i in thresholds:
                new_col = np.copy(col)
                new_col[col <= thresholds[i]] = True
                new_col[col > thresholds[i]] = False
                objects_bin.append(new_col)
            for i in thresholds:
                new_col = np.copy(col)
                new_col[col >= thresholds[i]] = True
                new_col[col < thresholds[i]] = False
                objects_bin.append(new_col)
        objects_bin = np.array(objects_bin, dtype=bool).T
        return objects_bin

    @staticmethod
    def rand_labels(rows, alpha, seed): # create random labels column
        rng = np.random.default_rng(seed)
        labels = np.zeros((rows,))
        end_of_1s = int(len(labels) * alpha)
        for i in range(end_of_1s):
            labels[i] = 1
        rng.shuffle(labels)
        return labels
    
    @staticmethod
    def rand_disc_df(rows, cols):
        data = np.random.randint(0, 10, size=(rows, cols))
        df = pd.DataFrame(data, columns=[f'x({i})' for i in range(cols)])
        return df

    @staticmethod
    def rand_cont_df(rows, cols):
        data = np.random.rand(rows, cols)
        df = pd.DataFrame(data, columns=[f'x({i})' for i in range(cols)])
        return df

    rand_disc_num_array = lambda rows, cols : np.random.randint(1, 10, (rows, cols))

    # rand_cont_num_array = lambda rows, cols : np.random.rand(rows, cols)

    # impact_obj = lambda ctx : lambda indices : (len(indices) / ctx.n) * (ctx.labels[indices].mean() - ctx.labels_mean)

    # impact_bnd = lambda ctx : lambda indices : (ctx.labels[indices].sum() / ctx.n) * (1 - ctx.labels_mean)

    # def impact_count_mean(labels):
    #     n = len(labels)
    #     m0 = sum(labels)/n

    #     def f(c, m):
    #         return c/n * (m - m0)

    #     return f

class NumContext:
    def __init__(self, objects):
        self.objects = objects
        self.n = len(objects)
        self.m = len(objects[0])

    def extension(self, intent: Intent): # get extension of interval pattern intent
        new_extension = np.arange(self.n)
        for j in range(self.m):
            low = intent[j][0]
            high = intent[j][1]
            mask = (self.objects[new_extension, j] >= low) & (self.objects[new_extension, j] <= high)
            new_extension = new_extension[mask]
        return new_extension

    def closure(self, extension):
        new_pattern = np.empty((self.m, 2))

        for j in range(self.m):
            new_pattern[j][0] = np.min(self.objects[extension], axis=0)[j] # get min value in attribute, set as lower threshold
            new_pattern[j][1] = np.max(self.objects[extension], axis=0)[j] # get max value in attribute, set as upper threshold

        return Intent(new_pattern)
        

class IPNode:
    def __init__(self, ctx, intent, active_attr=None, locked_attrs=None, shifted=False):
        self.ctx = ctx
        self.intent = intent
        self.extension = ctx.extension(intent)
        self.active_attr = active_attr
        self.locked_attrs = locked_attrs
        self.shifted = shifted

    def __repr__(self):
        repr_str = ""
        repr_str += "Intent: " + str(self.intent) + "\n"
        repr_str += "Active_attr: " + str(self.active_attr) + "\n"
        repr_str += "Obj val: " + str(self.obj_val) + "\n"
        repr_str += "Bound val: " + str(self.bnd_val) + "\n"
        return repr_str

    def get_minus_upper(self, j):
        new_pattern = np.copy(self.intent.pattern)
        new_pattern[j][1] -= 1
        new_intent = Intent(new_pattern)
        new_locked_attrs = np.copy(self.locked_attrs)
        new_locked_attrs[j] = True
        return IPNode(self.ctx, new_intent, j, new_locked_attrs, False)
    
    def get_plus_lower(self, j):
        new_pattern = np.copy(self.intent.pattern)
        new_pattern[j][0] += 1
        new_intent = Intent(new_pattern)
        return IPNode(self.ctx, new_intent, j, np.copy(self.locked_attrs), False)

    # bnds are inverted to allow for max heap
    def __le__(self, other):
        if -self.bnd_val == -other.bnd_val:
            return -self.obj_val <= -other.obj_val
        return -self.bnd_val <= -other.bnd_val

    def __eq__(self, other):
        return -self.bnd_val == -other.bnd_val

    def __ge__(self, other):
        if -self.bnd_val == -other.bnd_val:
            return -self.obj_val >= -other.obj_val
        return -self.bnd_val >= -other.bnd_val

    def __lt__(self, other):
        if -self.bnd_val == -other.bnd_val:
            return -self.obj_val < -other.obj_val
        return -self.bnd_val < -other.bnd_val

    def __gt__(self, other):
        if -self.bnd_val == -other.bnd_val:
            return -self.obj_val > -other.obj_val
        return -self.bnd_val > -other.bnd_val

class IPNode2:
    def __init__(self, ctx, intent, active_attr, locked_attrs, shifted):
        self.ctx = ctx
        self.intent = intent
        self.extension = ctx.extension(intent)
        self.active_attr = active_attr
        self.locked_attrs = locked_attrs
        self.shifted = shifted

    def __repr__(self):
        repr_str = ""
        repr_str += "Intent: " + str(self.intent) + "\n"
        repr_str += "Active_attr: " + str(self.active_attr) + "\n"
        repr_str += "Obj val: " + str(self.obj_val) + "\n"
        repr_str += "Bound val: " + str(self.bnd_val) + "\n"
        return repr_str

    def get_minus_upper(self, j):
        new_pattern = np.copy(self.intent.pattern)
        new_pattern[j][1] -= 1
        new_intent = Intent(new_pattern)
        new_locked_attrs = np.copy(self.locked_attrs)
        new_locked_attrs[j] = True
        return IPNode2(self.ctx, new_intent, j, new_locked_attrs, False)
    
    def get_plus_lower(self, j):
        new_pattern = np.copy(self.intent.pattern)
        new_pattern[j][0] += 1
        new_intent = Intent(new_pattern)
        return IPNode2(self.ctx, new_intent, j, np.copy(self.locked_attrs), False)

    # bnds are inverted to allow for max heap
    def __le__(self, other):
        return self.val <= other.val

    def __eq__(self, other):
        return self.val == other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __lt__(self, other):
        return self.val < other.val

    def __gt__(self, other):
        return self.val > other.val