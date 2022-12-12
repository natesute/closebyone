import numpy as np # type: ignore
from typing import Callable #type: ignore
import copy

class IPSearch:
    def __init__(self, root, curr_node, heap, context, res):
        self.root = root
        self.curr_node = curr_node
        self.heap = heap
        self.context = context
        self.res = res


class PropSearch:
    def __init__(self, curr_node, context, res):
        self.curr_node = curr_node
        self.context = context
        self.res = res


class Results:
    def __init__(self):
        self.num_candidates = 0
        self.num_nodes = 0
        self.max_obj = 0
        self.best_node = None
        self.nodes = []
        self.time = 0

    def __repr__(self):
        repr_str = ""
        repr_str += "Num Candidates: " + str(self.num_candidates) + "\n"
        repr_str += "Num Nodes: " + str(self.num_nodes) + "\n"
        repr_str += "Max Obj: " + str(self.max_obj) + "\n"
        repr_str += "Best Node: " + str(self.best_node) + "\n"
        repr_str += "Time: " + str(self.time) + "\n"
        return repr_str


class Intent:
    def __init__(self, pattern):
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

class Extent:
    def __init__(self, indices, objects): # objects = objects in extent
        self.indices = indices
        self.objects = objects
        self.m = (len(objects[0]) if len(objects) > 0 else 0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.indices[index]

    def get_closure(self):
        new_pattern = np.empty((self.m, 2))

        for j in range(self.m):
            new_pattern[j][0] = np.min(self.objects, axis=0)[j] # get min value in attribute, set as lower threshold
            new_pattern[j][1] = np.max(self.objects, axis=0)[j] # get max value in attribute, set as upper threshold

        return Intent(new_pattern)


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
        '''converts discrete numerical objectsset to binary'''
        objects_num = np.copy(old_objects_num)
        objects_bin = []
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
            # range through ordered threshold values
            for i in range(len(thresholds)):
                new_col = np.copy(col)
                # binarise column values according to proposition (>=)
                new_col[col >= thresholds[i]] = True
                new_col[col < thresholds[i]] = False
                objects_bin.append(new_col)
        # columns were appended to bin_d, so must be transposed to actually be columns
        objects_bin = np.array(objects_bin, dtype=bool).T
        return objects_bin

    def cont_num_to_bin(old_objects_num):
        '''converts continuous numerical objectsset to binary'''
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
    def rand_target_col(rows, alpha, seed): # create random target column
        rng = np.random.default_rng(seed)
        target_col = np.zeros((rows,))
        end_of_1s = int(len(target_col) * alpha)
        for i in range(end_of_1s):
            target_col[i] = 1
        rng.shuffle(target_col)
        return target_col

    @staticmethod
    def rand_disc_num_array(rows, cols):
        '''create random discrete numerical array'''
        objects_cols = np.random.randint(1, 10, (rows, cols))
        return objects_cols
    
    @staticmethod
    def rand_cont_num_array(rows, cols):
        '''create random continuous numerical array'''
        objects_cols = np.random.rand(rows, cols)
        return objects_cols

    @staticmethod
    def impact_obj(target):
        root_indices = np.arange(len(target))
        return lambda indices : (len(indices) / len(target)) * (Utilities.target_mean(target)(indices) - Utilities.target_mean(target)(root_indices))
    
    @staticmethod
    def impact_bnd(target):
        root_indices = np.arange(len(target))
        return lambda indices : (Utilities.target_sum(target)(indices) / len(target)) * (1 - Utilities.target_mean(target)(root_indices))

    @staticmethod
    def target_mean(target: np.ndarray) -> Callable[[np.ndarray], float]:
        '''returns a function that takes an array of indices and returns the mean of the target column at those indices'''
        return lambda indices : target[indices].mean()

    @staticmethod
    def target_sum(target: np.ndarray) -> Callable[[np.ndarray], float]:
        '''returns a function that takes an array of indices and returns the sum of the target column at those indices'''
        return lambda indices : target[indices].sum()


class Context:
    def __init__(self, target, objects, obj, bnd):
        self.objects = objects
        self.target = target
        self.obj = obj
        self.bnd = bnd
        self.get_target_mean = Utilities.target_mean(target)
        self.get_target_sum = Utilities.target_sum(target)
        self.n = len(target)
        self.m = len(objects[0])
        self.root_indices = np.arange(self.n)

    def get_extent(self, intent: Intent):
        new_indices = np.copy(self.root_indices)
        for j in range(self.m):
            low = intent[j][0]
            high = intent[j][1]
            mask = (self.objects[new_indices, j] >= low) & (self.objects[new_indices, j] <= high)
            new_indices = new_indices[mask]
        new_extent = Extent(new_indices, self.objects[new_indices])
        return new_extent

    def check_root(self):
        return np.add.reduce(self.objects, axis=0) == len(self.objects)

    def implied_on(self, j, extent):
        for i in extent:
            if not self.objects[i, j]:
                return False
        return True

class PropNode:
    def __init__(self, context, query, extent):
        self.context = context
        self.query = query
        self.extent = extent
    
    def __str__(self):
        return str(self.query)
        

class IPNode:
    def __init__(self, context, intent, active_attr, locked_attrs):
        self.context = context
        self.intent = intent
        self.extent = context.get_extent(intent)
        self.obj_val = context.obj(self.extent.indices)
        self.bnd_val = context.bnd(self.extent.indices)
        self.active_attr = active_attr
        self.locked_attrs = locked_attrs

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
        self.extent = self.context.get_extent(self.intent)
        new_locked_attrs = np.copy(self.locked_attrs)
        new_locked_attrs[j] = True
        return IPNode(self.context, new_intent, j, new_locked_attrs)
    
    def get_plus_lower(self, j):
        new_pattern = np.copy(self.intent.pattern)
        new_pattern[j][0] += 1
        new_intent = Intent(new_pattern)
        return IPNode(self.context, new_intent, j, np.copy(self.locked_attrs))

    # bnds are inverted to allow for max heap
    def __le__(self, other):
        return -self.bnd_val <= -other.bnd_val

    def __eq__(self, other):
        return -self.bnd_val == -other.bnd_val

    def __ge__(self, other):
        return -self.bnd_val >= -other.bnd_val

    def __lt__(self, other):
        return -self.bnd_val < -other.bnd_val

    def __gt__(self, other):
        return -self.bnd_val > -other.bnd_val