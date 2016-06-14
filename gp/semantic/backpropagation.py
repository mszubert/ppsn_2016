from collections import defaultdict
import random

import numpy

from gp.experiments import symbreg
from deap import gp


def get_node_depth(ind, node_index):
    stack = [0]
    for node in ind[:node_index]:
        depth = stack.pop()
        stack.extend([depth + 1] * node.arity)
    return stack.pop()


def rdo(ind, library_selector, target, node_selector, semantic_calculator, max_height=numpy.inf, inversion_map=None):
    ind.semantics = semantic_calculator(ind)
    index = node_selector(ind)
    node_desired_semantics = calculate_node_desired_semantics(ind, index, target, inversion_map)

    max_replacement_height = max_height - get_node_depth(ind, index)
    replacement, _ = library_selector(node_desired_semantics, max_height=max_replacement_height)

    slice_ = ind.searchSubtree(index)
    ind[slice_] = replacement[:]
    return ind,


def agx(ind1, ind2, library_selector, only_consistent=True):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    semantics1 = ind1.semantics[0]
    semantics2 = ind2.semantics[0]
    mid_point = (semantics1 + semantics2) / 2.0

    ind1 = rdo(ind1, library_selector, mid_point, only_consistent)[0]
    ind2 = rdo(ind2, library_selector, mid_point, only_consistent)[0]

    return ind1, ind2


def calculate_node_desired_semantics(ind, node_index, target_semantics, inversion_map=None):
    if inversion_map is None:
        inversion_map = get_numpy_inversion_map()

    children, parents = calculate_children_parents_indices(ind)

    root_node_path = [node_index]
    cur_node = node_index
    while cur_node > 0:
        cur_node = parents[cur_node]
        root_node_path.append(cur_node)
    root_node_path.reverse()

    root_desired_semantics = [[desired_value] for desired_value in target_semantics]
    desired_semantics = {0: root_desired_semantics}

    for path_index, path_node_index in enumerate(root_node_path[:-1]):
        node = ind[path_node_index]
        child_index = root_node_path[path_index + 1]
        arg_index = children[path_node_index].index(child_index)

        other_args_semantics = [ind.semantics[other_child] for other_child in children[path_node_index] if
                                other_child != child_index]
        child_semantics = calculate_desired_child_semantics(inversion_map[node.name], arg_index,
                                                            desired_semantics[path_node_index],
                                                            other_args_semantics)
        desired_semantics[child_index] = child_semantics

    return desired_semantics[node_index]


def calculate_desired_semantics(ind, target_semantics, inversion_map=None):
    if inversion_map is None:
        inversion_map = get_numpy_inversion_map()

    children = calculate_children_indices(ind)
    root_desired_semantics = [[desired_value] for desired_value in target_semantics]
    desired_semantics = {0: root_desired_semantics}

    for node_index, node in enumerate(ind):
        for arg_index, child_index in enumerate(children[node_index]):
            other_args_semantics = [ind.semantics[other_child] for other_child in children[node_index] if
                                    other_child != child_index]
            child_semantics = calculate_desired_child_semantics(inversion_map[node.name], arg_index,
                                                                desired_semantics[node_index],
                                                                other_args_semantics)
            desired_semantics[child_index] = child_semantics

    return desired_semantics


def calculate_children_indices(ind):
    stack = []
    children = defaultdict(list)
    for i, node in reversed(list(enumerate(ind))):
        for _ in range(node.arity):
            children[i].append(stack.pop())
        stack.append(i)
    return children


def calculate_children_parents_indices(ind):
    stack = []
    parents = {}
    children = defaultdict(list)
    for i, node in reversed(list(enumerate(ind))):
        for _ in range(node.arity):
            child = stack.pop()
            children[i].append(child)
            parents[child] = i
        stack.append(i)
    return children, parents


def calculate_desired_child_semantics(parent_inverter, child_index, desired_semantics, other_children_semantics):
    desired_child_semantics = []
    for fitness_case, desired_values in enumerate(desired_semantics):
        if desired_values == "*":
            desired_child_semantics.append("*")
            continue

        desired_child_semantics.append([])
        other_args_values = [other_args[fitness_case] for other_args in other_children_semantics]
        for desired_value in desired_values:
            inversions = parent_inverter(desired_value, child_index, other_args_values)
            if inversions == "*":
                desired_child_semantics[-1] = "*"
                break
            finite_inversions = filter(numpy.isfinite, inversions)
            desired_child_semantics[-1].extend(finite_inversions)
        if desired_child_semantics[-1] != "*" and len(desired_child_semantics[-1]) > 2:
            desired_child_semantics[-1] = random.sample(desired_child_semantics[-1], 2)

    return desired_child_semantics


def get_numpy_inversion_map():
    inversion_map = {numpy.add.__name__: lambda dv, _, ov: [dv - ov[0]],
                     numpy.subtract.__name__: lambda dv, index, ov: [dv + ov[0]] if index == 0 else [ov[0] - dv],
                     numpy.exp.__name__: lambda dv, _, __: [numpy.log(dv)] if dv >= 0.0 else [],
                     symbreg.numpy_protected_log_abs.__name__: lambda dv, _, __: [numpy.exp(dv), -numpy.exp(dv)]}

    def multiply_inversion(desired_value, _, other_values):
        if not numpy.isclose(other_values[0], 0.0):
            return [desired_value / other_values[0]]
        return "*" if numpy.isclose(desired_value, 0.0) else []

    inversion_map[numpy.multiply.__name__] = multiply_inversion

    def division_dividend_inversion(desired_value, index, other_values):
        if index == 0:
            if numpy.isclose(other_values[0], 0.0):
                return [desired_value]
            if numpy.isfinite(other_values[0]):
                return [desired_value * other_values[0]]
            return "*" if numpy.isclose(desired_value, 0.0) else []
        else:
            if numpy.isclose(desired_value, 0.0):
                return "*" if numpy.isclose(other_values[0], 0.0) else []
            if not numpy.isclose(other_values[0], 0.0):
                return [other_values[0] / desired_value]
            return []

    inversion_map[symbreg.numpy_protected_div_dividend.__name__] = division_dividend_inversion

    def sin_inversion(desired_value, _, __):
        if numpy.abs(desired_value) <= 1.0:
            asin = numpy.arcsin(desired_value)
            return [asin, asin - 2 * numpy.pi] if asin > 0 else [asin, asin + 2 * numpy.pi]
        else:
            return []

    inversion_map[numpy.sin.__name__] = sin_inversion

    def cos_inversion(desired_value, _, __):
        if numpy.abs(desired_value) <= 1.0:
            acos = numpy.arccos(desired_value)
            return [acos, acos - 2 * numpy.pi]
        else:
            return []

    inversion_map[numpy.cos.__name__] = cos_inversion

    return inversion_map


def library_search(desired_semantics, lib, distance_measure, max_height=numpy.inf, check_constant_prob=1.0):
    num_trees = len(lib.trees) if max_height >= len(lib.height_levels) else lib.height_levels[max_height]

    single_value_indices, multi_valued_indices = [], []
    single_valued_vector, multi_valued_vector = [], []
    candidate_constant_values = set()
    for index, desired_values in enumerate(desired_semantics):
        if len(desired_values) == 1 and desired_values != "*":
            single_valued_vector.append(desired_values[0])
            candidate_constant_values.add(desired_values[0])
            single_value_indices.append(index)
        elif len(desired_values) > 1:
            candidate_constant_values.update(desired_values)
            multi_valued_indices.append(index)
            multi_valued_vector.append(desired_values)

    if len(single_value_indices) + len(multi_valued_indices) == 0:
        tree_index = random.randrange(num_trees)
        return lib.trees[tree_index][:], lib.semantic_array[tree_index]

    distances = distance_measure(lib.semantic_array[:num_trees, single_value_indices], single_valued_vector, axis=1)
    for multi_valued_index in multi_valued_indices:
        diff = lib.semantic_array[:num_trees, multi_valued_index, None] - desired_semantics[multi_valued_index]
        distances += numpy.nanmin(numpy.abs(diff), axis=1)

    min_distance_index = numpy.nanargmin(distances)
    if random.random() < check_constant_prob:
        min_constant_distance = numpy.inf
        min_constant_value = None
        for constant_value in candidate_constant_values:
            constant_distance = numpy.sum(numpy.abs(numpy.subtract(single_valued_vector, constant_value)))
            for multi_values in multi_valued_vector:
                diff = numpy.subtract(multi_values, constant_value)
                multi_value_distance = numpy.nanmin(numpy.abs(diff))
                constant_distance += multi_value_distance
            if constant_distance < min_constant_distance:
                min_constant_distance = constant_distance
                min_constant_value = constant_value

        if min_constant_distance < distances[min_distance_index]:
            constant = min_constant_value.item()
            return [gp.Terminal(constant, False, float)], [constant] * len(desired_semantics)

    return lib.trees[min_distance_index][:], lib.semantic_array[min_distance_index]
