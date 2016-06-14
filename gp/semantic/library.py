import random
import operator

import cachetools
from deap import gp
from deap.gp import PrimitiveTree
import numpy

from gp.semantic import semantics


def get_closest_decorator(get_closest_func):
    def get_closest_wrapped(self, vector, *args, **kwargs):
        if not self.trees:
            value = random.choice(vector)
            return [gp.Terminal(value, False, float)], numpy.array([value] * len(vector))
        if not numpy.all(numpy.isfinite(vector)):
            max_height = kwargs.get("max_height", numpy.inf)
            num_trees = len(self.trees) if max_height >= len(self.height_levels) else self.height_levels[max_height]
            tree_index = random.randrange(num_trees)
            return self.trees[tree_index][:], self.semantic_array[tree_index]
        if is_constant(vector):
            return [gp.Terminal(vector[0], False, float)], numpy.array([vector[0]] * len(vector))
        return get_closest_func(self, vector, *args, **kwargs)

    return get_closest_wrapped


class SemanticLibrary(object):
    def __init__(self):
        self.trees = []
        self.height_levels = []
        self.semantic_array = []
        self.unique_semantics = []
        self.unit_semantic_array = []

    def generate_random_trees(self, pset, predictors, num_trees, generator):
        sem_dictionary = {}

        for terminal in pset.terminals[pset.ret]:
            tree = [terminal]
            tree_semantics = semantics.calculate_semantics(tree, pset.context, predictors)
            root_semantics = tree_semantics[0]
            if numpy.isfinite(numpy.sum(root_semantics)) and not is_constant(root_semantics):
                if root_semantics.tostring() not in sem_dictionary:
                    sem_dictionary[root_semantics.tostring()] = root_semantics

        expression_dict = cachetools.LRUCache(maxsize=10000)
        while len(sem_dictionary) < num_trees:
            tree = generator()
            tree_semantics = semantics.calculate_semantics(tree, pset.context, predictors,
                                                           expression_dict=expression_dict)
            root_semantics = tree_semantics[0]
            if numpy.isfinite(numpy.sum(root_semantics)) and not is_constant(root_semantics):
                if root_semantics.tostring() not in sem_dictionary:
                    sem_dictionary[root_semantics.tostring()] = root_semantics
                elif sem_dictionary[root_semantics.tostring()].tree_size > root_semantics.tree_size:
                    sem_dictionary[root_semantics.tostring()] = root_semantics

        self.unique_semantics = sem_dictionary.values()
        self.unique_semantics.sort(key=operator.attrgetter("tree_height"))

        self.height_levels = calculate_height_levels(self.unique_semantics)
        self.trees = [s.get_nodes() for s in self.unique_semantics]
        self.semantic_array = numpy.array(self.unique_semantics)
        self.unit_semantic_array = numpy.array([calc_unit_vector(s) for s in self.unique_semantics])

    def generate_trees(self, pset, depth, predictors):
        self.trees = generate_and_evaluate_semantically_distinct_trees(pset, depth, predictors)
        self.trees = filter(lambda t: not is_constant(t.semantics), self.trees)
        self.height_levels = calculate_height_levels(self.trees)
        self.unique_semantics = [tree.semantics for tree in self.trees]
        self.semantic_array = numpy.array(self.unique_semantics)
        self.unit_semantic_array = numpy.array([calc_unit_vector(tree.semantics) for tree in self.trees])

    @get_closest_decorator
    def get_closest(self, vector, distance_measure, max_height=numpy.inf, k=1, check_constant_prob=1.0):
        if max_height >= len(self.height_levels):
            dists = distance_measure(self.semantic_array, vector, axis=1)
        else:
            dists = distance_measure(self.semantic_array[:self.height_levels[max_height]], vector, axis=1)

        if 1 < k < len(dists):
            indices = numpy.argpartition(dists, k)
            min_distance_index = numpy.random.choice(indices[:k])
        else:
            min_distance_index = numpy.nanargmin(dists)

        if random.random() < check_constant_prob:
            constant = numpy.median(vector).item()
            constant_distance = distance_measure(vector, constant)
            if constant_distance < dists[min_distance_index]:
                return [gp.Terminal(constant, False, float)], numpy.array([constant] * len(vector))

        return self.trees[min_distance_index][:], self.semantic_array[min_distance_index]

    def get_closest_inconsistencies(self, vector, distance_measure, max_height=numpy.inf, check_constant_prob=1.0):
        num_trees = len(self.trees) if max_height >= len(self.height_levels) else self.height_levels[max_height]

        consistent_indices = ~numpy.isnan(vector)
        if not numpy.any(consistent_indices):
            tree_index = random.randrange(num_trees)
            return self.trees[tree_index][:], self.semantic_array[tree_index]
        consistent_vector = vector[consistent_indices]

        if not numpy.all(numpy.isfinite(consistent_vector)):
            tree_index = random.randrange(num_trees)
            return self.trees[tree_index][:], self.semantic_array[tree_index]
        if is_constant(consistent_vector):
            return [gp.Terminal(consistent_vector[0], False, float)], [consistent_vector[0]] * len(vector)

        dists = distance_measure(self.semantic_array[:num_trees, consistent_indices], consistent_vector, axis=1)
        min_distance_index = numpy.nanargmin(dists)

        if random.random() < check_constant_prob:
            constant = numpy.median(consistent_vector).item()
            constant_distance = distance_measure(consistent_vector, constant)
            if constant_distance < dists[min_distance_index]:
                return [gp.Terminal(constant, False, float)], [constant] * len(vector)

        return self.trees[min_distance_index][:], self.semantic_array[min_distance_index]

    def get_closest_direction_scaled_inconsistencies(self, vector, pset, distance_measure, max_height=numpy.inf,
                                                     constant_prob=1.0):
        num_trees = len(self.trees) if max_height >= len(self.height_levels) else self.height_levels[max_height]

        consistent_indices = ~numpy.isnan(vector)
        if not numpy.any(consistent_indices):
            tree_index = random.randrange(num_trees)
            return self.trees[tree_index][:], self.semantic_array[tree_index]
        consistent_vector = vector[consistent_indices]

        if not numpy.all(numpy.isfinite(consistent_vector)):
            tree_index = random.randrange(num_trees)
            return self.trees[tree_index][:], self.semantic_array[tree_index]
        if is_constant(consistent_vector):
            return [gp.Terminal(consistent_vector[0], False, float)], [consistent_vector[0]] * len(vector)

        unit_vector, vector_norm = calc_unit_vector_norm(consistent_vector)
        dot_products = numpy.dot(self.unit_semantic_array[:num_trees, consistent_indices], unit_vector)
        index = numpy.nanargmax(dot_products)

        best_tree = self.trees[index]
        best_tree_semantics = self.semantic_array[index]
        best_tree_norm = numpy.sqrt(best_tree_semantics.dot(best_tree_semantics))
        constant_value = round((vector_norm * dot_products[index]) / best_tree_norm, 3)

        if random.random() < constant_prob:
            distance = distance_measure(self.semantic_array[index] * constant_value, vector)
            constant = numpy.median(consistent_vector).item()
            constant_distance = distance_measure(consistent_vector, constant)
            if constant_distance < distance:
                return [gp.Terminal(constant, False, float)], numpy.array([constant] * len(vector))

        constant_terminal = gp.Terminal(constant_value, False, float)
        return [pset.mapping["multiply"], constant_terminal] + best_tree[:], self.semantic_array[index] * constant_value

    @get_closest_decorator
    def get_closest_direction(self, vector, distance_measure, max_height=numpy.inf, k=1, check_constant=False):
        unit_vector = calc_unit_vector(vector)

        if max_height > len(self.height_levels):
            dot_products = numpy.dot(self.unit_semantic_array, unit_vector)
        else:
            dot_products = numpy.dot(self.unit_semantic_array[:self.height_levels[max_height]], unit_vector)

        if k > 1:
            indices = numpy.argpartition(dot_products, -k)
            index = numpy.random.choice(indices[-k:])
        else:
            index = numpy.nanargmax(dot_products)

        print numpy.arccos(dot_products[index])
        print unit_vector
        print self.unit_semantic_array[index]

        if check_constant:
            distance = distance_measure(self.semantic_array[index], vector)
            constant = numpy.median(vector).item()
            constant_distance = distance_measure(vector, constant)
            if constant_distance < distance:
                return [gp.Terminal(constant, False, float)], numpy.array([constant] * len(vector))

        return self.trees[index][:], self.semantic_array[index]

    @get_closest_decorator
    def get_closest_direction_scaled(self, vector, distance_measure, pset, max_height=numpy.inf, k=1,
                                     check_constant_prob=1.0):
        unit_vector, vector_norm = calc_unit_vector_norm(vector)
        if max_height >= len(self.height_levels):
            dot_products = numpy.dot(self.unit_semantic_array, unit_vector)
        else:
            dot_products = numpy.dot(self.unit_semantic_array[:self.height_levels[max_height]], unit_vector)

        if k > 1:
            indices = numpy.argpartition(dot_products, -k)
            index = numpy.random.choice(indices[-k:])
        else:
            index = numpy.nanargmax(dot_products)

        best_tree = self.trees[index]
        best_tree_semantics = self.semantic_array[index]
        best_tree_norm = numpy.sqrt(best_tree_semantics.dot(best_tree_semantics))
        constant_value = round((vector_norm * dot_products[index]) / best_tree_norm, 3)

        if random.random() < check_constant_prob:
            distance = distance_measure(self.semantic_array[index] * constant_value, vector)
            constant = numpy.median(vector).item()
            constant_distance = distance_measure(vector, constant)
            if constant_distance < distance:
                return [gp.Terminal(constant, False, float)], numpy.array([constant] * len(vector))

        constant_terminal = gp.Terminal(constant_value, False, float)
        return [pset.mapping["multiply"], constant_terminal] + best_tree[:], self.semantic_array[index] * constant_value


class PopulationLibrary(SemanticLibrary):
    def __init__(self, reset=True, tree_size_limit=numpy.inf, semantic_calculator=None):
        super(PopulationLibrary, self).__init__()
        self.reset = reset
        self.tree_size_limit = tree_size_limit
        self.height_levels = []
        self.unique_semantics = []
        self.semantic_dictionary = dict()
        self.semantics_calculator = semantic_calculator

    def update(self, population):
        if self.reset:
            self.semantic_dictionary = dict()

        if self.semantics_calculator is not None:
            for ind in population:
                ind.semantics = self.semantics_calculator(ind)

        for ind in population:
            for subtree_semantics in ind.semantics:
                if subtree_semantics.tree_size <= self.tree_size_limit and numpy.isfinite(
                        numpy.sum(subtree_semantics)) and not is_constant(subtree_semantics):
                    if subtree_semantics.tostring() not in self.semantic_dictionary:
                        self.semantic_dictionary[subtree_semantics.tostring()] = subtree_semantics
                    elif self.semantic_dictionary[subtree_semantics.tostring()].tree_size > subtree_semantics.tree_size:
                        self.semantic_dictionary[subtree_semantics.tostring()] = subtree_semantics

        self.unique_semantics = self.semantic_dictionary.values()
        self.unique_semantics.sort(key=operator.attrgetter("tree_height"))

        self.height_levels = calculate_height_levels(self.unique_semantics)
        self.trees = [s.get_nodes() for s in self.unique_semantics]
        self.semantic_array = numpy.array(self.unique_semantics)
        self.unit_semantic_array = numpy.array([calc_unit_vector(s) for s in self.unique_semantics])

    def save(self, log_file):
        pass


def calculate_height_levels(sorted_semantics):
    height_levels = []
    current_height = 0
    for i, s in enumerate(sorted_semantics):
        if s.tree_height > current_height:
            height_levels.extend([i] * (s.tree_height - current_height))
            current_height = s.tree_height
    return height_levels


def is_constant(vector):
    return numpy.ptp(vector) <= 10e-8


def calc_unit_vector(vector):
    norm = numpy.sqrt(vector.dot(vector))
    return vector / norm


def calc_unit_vector_norm(vector):
    norm = numpy.sqrt(vector.dot(vector))
    return vector / norm, norm


def generate_and_evaluate_semantically_distinct_trees(pset, max_depth, predictors):
    last_trees = []
    smaller_trees = []
    sem_dictionary = {}

    for terminal in pset.terminals[pset.ret]:
        tree = PrimitiveTree([terminal])
        tree.semantics = semantics.get_terminal_semantics(terminal, pset.context, predictors)
        if numpy.isfinite(numpy.sum(tree.semantics)) and tree.semantics.tostring() not in sem_dictionary:
            sem_dictionary[tree.semantics.tostring()] = tree
            tree.tree_height = 0
            last_trees.append(tree)

    for depth in range(max_depth):
        trees = []
        for function in pset.primitives[pset.ret]:
            func = pset.context[function.name]
            if function.arity == 1:
                for subtree in last_trees:
                    tree = PrimitiveTree([function] + subtree[:])
                    tree.semantics = func(subtree.semantics)
                    if numpy.isfinite(numpy.sum(tree.semantics)) and tree.semantics.tostring() not in sem_dictionary:
                        sem_dictionary[tree.semantics.tostring()] = tree
                        tree.tree_height = depth + 1
                        trees.append(tree)
            elif function.arity == 2:
                for subtree_1 in last_trees:
                    for subtree_2 in last_trees:
                        tree = PrimitiveTree([function] + subtree_1[:] + subtree_2[:])
                        tree.semantics = func(subtree_1.semantics, subtree_2.semantics)
                        if numpy.isfinite(
                                numpy.sum(tree.semantics)) and tree.semantics.tostring() not in sem_dictionary:
                            sem_dictionary[tree.semantics.tostring()] = tree
                            tree.tree_height = depth + 1
                            trees.append(tree)

                    for subtree_2 in smaller_trees:
                        tree = PrimitiveTree([function] + subtree_1[:] + subtree_2[:])
                        tree.semantics = func(subtree_1.semantics, subtree_2.semantics)
                        if numpy.isfinite(
                                numpy.sum(tree.semantics)) and tree.semantics.tostring() not in sem_dictionary:
                            sem_dictionary[tree.semantics.tostring()] = tree
                            tree.tree_height = depth + 1
                            trees.append(tree)
                        tree = PrimitiveTree([function] + subtree_2[:] + subtree_1[:])
                        tree.semantics = func(subtree_2.semantics, subtree_1.semantics)
                        if numpy.isfinite(
                                numpy.sum(tree.semantics)) and tree.semantics.tostring() not in sem_dictionary:
                            sem_dictionary[tree.semantics.tostring()] = tree
                            tree.tree_height = depth + 1
                            trees.append(tree)

        smaller_trees.extend(last_trees)
        last_trees = trees

    return smaller_trees + last_trees
