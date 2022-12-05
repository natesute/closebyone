from heapq import heappop, heappush
import numpy as np


class Results:
    def __init__(self):
        self.num_candidates = 0
        self.num_nodes = 0
        self.max_obj = 0

    def __repr__(self):
        repr_str = ""
        repr_str += "Num Candidates: " + str(self.num_candidates) + "\n"
        repr_str += "Num Nodes: " + str(self.num_nodes) + "\n"
        repr_str += "Max Obj: " + str(self.max_obj) + "\n"
        return repr_str

class Utilities:

    @staticmethod
    def rand_target_col(rows, alpha, seed): # create random target column
        rng = np.random.default_rng(seed)
        target_col = np.zeros((rows, 1))
        end_of_1s = int(len(target_col) * alpha)
        for i in range(end_of_1s):
            target_col[i] = 1
        rng.shuffle(target_col)
        return target_col

    @staticmethod
    def rand_disc_num_array(rows, cols): # create random discrete numerical array
        data_cols = np.random.randint(1, 10, (rows, cols))
        return data_cols

    @staticmethod
    def impact_obj(target):
        root_indices = np.arange(len(target))
        return lambda indices : (len(indices) / len(target)) * (Utilities.target_mean(target)(indices) - Utilities.target_mean(target)(root_indices))
    
    @staticmethod
    def impact_bnd(target):
        root_indices = np.arange(len(target))
        return lambda indices : (Utilities.target_sum(target)(indices) / len(target)) * (1 - Utilities.target_mean(target)(root_indices))

    @staticmethod
    def target_mean(target):
        return lambda indices : target[indices].mean()

    @staticmethod
    def target_sum(target):
        return lambda indices : target[indices].sum()

class Context:
    def __init__(self, target, objects, obj, bnd):
        self.objects = objects
        self.target = target
        self.obj = obj
        self.bnd = bnd
        self.target_mean = Utilities.target_mean(target)
        self.target_sum = Utilities.target_sum(target)
        self.n = len(target)
        self.m = len(objects[0])
        self.root_indices = np.arange(self.n)


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

    def get_minus_upper(self, j):
        new_pattern = np.copy(self.pattern)
        new_pattern[j][1] -= 1
        return Intent(new_pattern)
    
    def get_plus_lower(self, j):
        new_pattern = np.copy(self.pattern)
        new_pattern[j][0] += 1
        return Intent(new_pattern)
    
    def fully_closed(self, j):
        return self.pattern[j][0] == self.pattern[j][1]


class Extent:
    def __init__(self, indices, objects): # objects = objects in extent
        self.indices = indices
        self.objects = objects
        self.m = len(objects[0])

    def __len__(self):
        return len(self.indices)

    def get_closure(self):
        new_pattern = np.empty((self.m, 2))

        for j in range(self.m):
            new_pattern[j][0] = np.min(self.objects, axis=0)[j] # get min value in attribute, set as lower threshold
            new_pattern[j][1] = np.max(self.objects, axis=0)[j] # get max value in attribute, set as upper threshold

        return Intent(new_pattern)


class Node:
    def __init__(self, extent, intent, obj_val, bnd_val, locked_attrs, active_attr):
        self.extent = extent
        assert type(intent) == Intent, "node intent is not object"
        self.intent = intent
        self.obj_val = obj_val
        self.bnd_val = bnd_val
        self.locked_attrs = locked_attrs
        self.active_attr = active_attr

    def __repr__(self):
        repr_str = ""
        repr_str += "Intent: " + str(self.intent) + "\n"
        repr_str += "Active_attr: " + str(self.active_attr) + "\n"
        repr_str += "Obj val: " + str(self.obj_val) + "\n"
        repr_str += "Bound val: " + str(self.bnd_val) + "\n"
        return repr_str

    def __le__(self, other):
        return self.bnd_val <= other.bnd_val

    def __eq__(self, other):
        return self.bnd_val == other.bnd_val

    def __ge__(self, other):
        return self.bnd_val >= other.bnd_val

    def __lt__(self, other):
        return self.bnd_val < other.bnd_val

    def __gt__(self, other):
        return self.bnd_val > other.bnd_val


class BFS: # best first search
    def __init__(self, curr_node, heap, context, res):
        self.curr_node = curr_node
        self.heap = heap
        self.context = context
        self.res = res

    @staticmethod
    def is_canonical(curr_intent, new_intent, j):
        for i in range(j + 1, len(curr_intent[0])):
            # if a bound has been changed in a previous (greater than current j) attribute
            if curr_intent[i][0] != new_intent[i][0] or curr_intent[i][1] != new_intent[i][1]:
                return False
        return True

    def get_extent(self, intent):
        new_indices = np.copy(self.context.root_indices)
        for j in range(self.context.m):
            low = intent[j][0]
            high = intent[j][1]
            mask = (self.context.objects[new_indices, j] >= low) & (self.context.objects[new_indices, j] <= high)
            new_indices = new_indices[mask]
        new_extent = Extent(new_indices, self.context.objects[new_indices])
        return new_extent

    def push_children(self, j):
        if not self.curr_node.intent.fully_closed(j):
            new_locked_attrs = np.copy(self.curr_node.locked_attrs)
            new_locked_attrs[j] = 1
            new_intent = self.curr_node.intent.get_minus_upper(j)
            new_extent = self.get_extent(new_intent)
            new_obj_val = self.context.obj(self.curr_node.extent.indices)
            new_bnd_val = self.context.bnd(self.curr_node.extent.indices)

            heappush(self.heap, Node(new_extent, new_intent, new_obj_val, new_bnd_val, new_locked_attrs, j))

            if not self.curr_node.locked_attrs[j]: # if j is not a locked attribute
                new_intent = self.curr_node.intent.get_plus_lower(j)
                new_extent = self.get_extent(new_intent)
                new_obj_val = self.context.obj(new_extent.indices)
                new_bnd_val = self.context.bnd(new_extent.indices)
                heappush(self.heap, Node(new_extent, new_intent, new_obj_val, new_bnd_val, self.curr_node.locked_attrs, j))

    def run(self, root_node):
        max_bnd = self.context.bnd(root_node.extent.indices)
        heappush(self.heap, root_node)
        
        while self.heap: # while queue is not empty
            self.curr_node = heappop(self.heap)
            
            j = self.curr_node.active_attr
            max_obj = max(self.curr_node.obj_val, self.res.max_obj)
            if self.context.obj(self.curr_node.extent.indices) > self.res.max_obj or self.curr_node == root_node: # root node check because obj > max_obj check fails on root
                closed_intent = self.curr_node.extent.get_closure()
                if self.is_canonical(self.curr_node.intent, closed_intent, j):
                    self.curr_node.intent = closed_intent
                    self.res.num_nodes += 1
                    print(self.curr_node, end="\n\n")
                    if max_obj == max_bnd:
                        break
                    self.push_children(j)
                    if j>0:
                        self.push_children(j-1)