from heapq import heappop, heappush
import numpy as np

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

    def minus_upper(self, j):
        new_pattern = np.copy(self.pattern)
        new_pattern[j][1] -= 1
        return new_pattern
    
    def plus_lower(self, j):
        new_pattern = np.copy(self.pattern)
        new_pattern[j][0] += 1
        return new_pattern
    
    def fully_closed(self, j):
        return self.pattern[j][0] == self.pattern[j][1]

class Node:
    def __init__(self, extent, intent, obj, bnd, locked_attrs, active_attr):
        self.extent = extent
        self.intent = intent
        self.obj = obj
        self.bnd = bnd
        self.locked_attrs = locked_attrs
        self.active_attr = active_attr

    def __repr__(self):
        return str(self.intent)

    def __le__(self, other):
        return self.bnd <= other.bnd

    def __eq__(self, other):
        return self.bnd == other.bnd

    def __ge__(self, other):
        return self.bnd >= other.bnd

    def __lt__(self, other):
        return self.bnd < other.bnd

    def __gt__(self, other):
        return self.bnd > other.bnd


class CloseByOneBFS:
    def __init__(self, target, data, obj, bnd):
        self.target = target
        self.data = data
        self.f = obj
        self.g = bnd
        self.m = len(data[0])
        self.n = len(data)

    def get_mins(self, extent):
        data_in_extent = self.data[extent]
        return np.min(data_in_extent, axis=0)

    def get_maxs(self, extent):
        data_in_extent = self.data[extent]
        return np.max(data_in_extent, axis=0)

    def closure_of(self, extent):
        new_pattern = np.empty((self.m, 2))
        mins = self.get_mins(extent)
        maxs = self.get_maxs(extent)

        for j in range(self.m):
            new_pattern[j][0] = mins[j]
            new_pattern[j][1] = maxs[j]

        return Intent(new_pattern)

    def is_canonical(self, curr_node, new_intent):
        intent = curr_node.intent
        j = curr_node.active_attr
        for i in range(j + 1, len(curr_node.intent[0])):
            # if a bound has been changed in a previous (greater than current j) attribute
            if intent[i][0] != new_intent[i][0] or intent[i][1] != new_intent[i][1]:
                return False
        return True

    def run(self, root_node):
        max_bnd = self.g(root_node.extent)
        num_nodes = 0
        max_obj = 0
        heap = []
        heappush(heap, root_node)
        
        while heap: # while queue is not empty
            curr_node = heappop(heap)
            j = curr_node.active_attr
            print(curr_node, end=" ")
            max_obj = max(curr_node.obj, max_obj)
            if max_obj == max_bnd:
                break
            if self.g(curr_node.extent) > max_obj:
                closed_intent = self.closure_of(curr_node.extent)
                if self.is_canonical(curr_node, closed_intent):
                    curr_node.intent = closed_intent
                    num_nodes += 1
                    if not curr_node.intent.fully_closed(j):
                        new_locked_attrs = np.copy(curr_node.locked_attrs)
                        new_locked_attrs[j] = 1
                        heappush(heap, Node(curr_node.extent, curr_node.intent.minus_upper(j), self.f(curr_node.extent), self.g(curr_node.extent), new_locked_attrs, j))

                        if curr_node.locked_attrs[j] != 0:
                            heappush(heap, Node(curr_node.extent, curr_node.intent.plus_lower(j), self.f(curr_node.extent), self.g(curr_node.extent), curr_node.locked_attrs, j))
                    if j:
                        heappush(heap, Node(curr_node.extent, curr_node.intent, self.f(curr_node.extent), self.g(curr_node.extent), curr_node.locked_attrs, j-1))
        print()