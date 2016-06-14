import copy

import cachetools
import numpy
from deap import gp


class SemanticPrimitiveTree(gp.PrimitiveTree):
    def __init__(self, content):
        gp.PrimitiveTree.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        old_semantics = getattr(self, "semantics", None)
        self.__dict__["semantics"] = None
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        new.__dict__["semantics"] = self.__dict__["semantics"] = old_semantics
        return new


class Semantics(numpy.ndarray):
    def __new__(cls, numpy_array, node_index, expr, ind, tree_size=1, tree_height=0):
        self = numpy_array.view(cls)
        self.node_index = node_index
        self.tree_size = tree_size
        self.tree_height = tree_height
        self.expr = expr
        self.ind = ind
        return self

    def get_nodes(self):
        return self.ind[self.node_index:self.node_index + self.tree_size]

    def __deepcopy__(self, memo):
        copy = numpy.copy(self)
        return type(self)(copy, self.node_index, self.expr, self.ind, self.tree_size, self.tree_height)


def calc_eval_semantics(ind, context, predictors, eval_semantics, expression_dict=None, arg_prefix="ARG"):
    ind.semantics = calculate_semantics(ind, context, predictors, expression_dict, arg_prefix)
    return eval_semantics(ind.semantics[0])


def calculate_semantics(ind, context, predictors, expression_dict=None, arg_prefix="ARG"):
    semantics = []
    sizes_stack = []
    height_stack = []
    semantics_stack = []
    expressions_stack = []

    if expression_dict is None:
        expression_dict = cachetools.LRUCache(maxsize=100)

    for index, node in enumerate(reversed(ind)):
        expression = node.format(*[expressions_stack.pop() for _ in range(node.arity)])
        subtree_semantics = [semantics_stack.pop() for _ in range(node.arity)]
        subtree_size = sum([sizes_stack.pop() for _ in range(node.arity)]) + 1
        subtree_height = max([height_stack.pop() for _ in range(node.arity)]) + 1 if node.arity > 0 else 0

        if expression in expression_dict:
            vector = expression_dict[expression]
        else:
            vector = get_node_semantics(node, subtree_semantics, predictors, context, arg_prefix)
            expression_dict[expression] = vector

        expressions_stack.append(expression)
        semantics_stack.append(vector)
        sizes_stack.append(subtree_size)
        height_stack.append(subtree_height)

        semantics.append(Semantics(vector, len(ind) - index - 1, expression, ind, subtree_size, subtree_height))

    semantics.reverse()
    return semantics


def get_node_semantics(node, subtree_semantics, predictors, context, arg_prefix="ARG"):
    if isinstance(node, gp.Terminal):
        vector = get_terminal_semantics(node, context, predictors, arg_prefix)
    else:
        with numpy.errstate(over='ignore', divide='ignore', invalid='ignore'):
            vector = context[node.name](*subtree_semantics)
    return vector


def get_terminal_semantics(node, context, predictors, arg_prefix="ARG"):
    if isinstance(node, gp.Ephemeral) or isinstance(node.value, float):
        return numpy.ones(len(predictors)) * node.value

    if node.value in context:
        return numpy.ones(len(predictors)) * context[node.value]

    arg_index = node.value[len(arg_prefix):]
    return predictors[:, int(arg_index)]


def get_unique_semantics(semantics, max_diff, distance_measure):
    unique_semantics = []
    semantic_array = numpy.array(semantics)
    unique = numpy.ones(shape=(len(semantics),))

    groups = []
    for index, sem in enumerate(semantics):
        if unique[index] > 0:
            distances = distance_measure(semantic_array[index + 1:, :], semantic_array[index], axis=1)
            zero_indices = numpy.where(distances < max_diff)
            groups.append(len(zero_indices[0]) + 1)
            unique[zero_indices[0] + index + 1] = 0.0
            unique_semantics.append(sem)
    return unique_semantics, groups
