import numpy

from gp.experiments import symbreg


def get_numpy_operators_map(operators):
    operators_map = {numpy.add.__name__: lambda target, current: target - current,
                     numpy.subtract.__name__: lambda target, current: target + current,
                     symbreg.numpy_protected_div_dividend.__name__: lambda target, current: target * current,
                     numpy.multiply.__name__: lambda target, current: symbreg.numpy_protected_div_one(target, current)}

    return {op.__name__: operators_map[op.__name__] for op in operators}


def get_all_numpy_operations(pset):
    add, subtract = pset.mapping["add"], pset.mapping["subtract"]
    multiply, divide = pset.mapping["multiply"], pset.mapping["numpy_protected_div_dividend"]

    binary_funcs = [(lambda subtree_semantics, target: target - subtree_semantics,
                     lambda subtree_semantics, lib_semantics: subtree_semantics + lib_semantics,
                     lambda subtree, lib_proc: [add] + lib_proc[:] + subtree[:]),

                    (lambda subtree_semantics, target: target + subtree_semantics,
                     lambda subtree_semantics, lib_semantics: lib_semantics - subtree_semantics,
                     lambda subtree, lib_proc: [subtract] + lib_proc[:] + subtree[:]),

                    (lambda subtree_semantics, target: subtree_semantics - target,
                     lambda subtree_semantics, lib_semantics: subtree_semantics - lib_semantics,
                     lambda subtree, lib_proc: [subtract] + subtree[:] + lib_proc[:]),

                    (lambda subtree_semantics, target: symbreg.numpy_protected_div_one(target, subtree_semantics),
                     lambda subtree_semantics, lib_semantics: subtree_semantics * lib_semantics,
                     lambda subtree, lib_proc: [multiply] + subtree[:] + lib_proc[:]),

                    (lambda subtree_semantics, target: symbreg.numpy_protected_div_one(subtree_semantics, target),
                     lambda subtree_semantics, lib_semantics: symbreg.numpy_protected_div_dividend(subtree_semantics,
                                                                                                   lib_semantics),
                     lambda subtree, lib_proc: [divide] + subtree[:] + lib_proc[:]),

                    (lambda subtree_semantics, target: subtree_semantics * target,
                     lambda subtree_semantics, lib_semantics: symbreg.numpy_protected_div_dividend(lib_semantics,
                                                                                                   subtree_semantics),
                     lambda subtree, lib_proc: [divide] + lib_proc[:] + subtree[:])
                    ]
    return binary_funcs


def forward_propagation_mutation_all(ind, pset, library_selector, semantic_calculator, target, binary_operations,
                                     unary_operators, distance_measure, node_selector=None):
    if node_selector is None:
        index = 0
    else:
        index = node_selector(ind)

    subtree = ind[ind.searchSubtree(index)]
    subtree_semantics = semantic_calculator(subtree)[0]
    if not numpy.isfinite(numpy.sum(subtree_semantics)):
        return ind,

    offspring = None
    min_error = numpy.inf
    for i, (residual_func, output_func, tree_func) in enumerate(binary_operations):
        residual = residual_func(subtree_semantics, target)
        library_procedure, procedure_semantics = library_selector(residual)
        output = output_func(subtree_semantics, procedure_semantics)
        error = distance_measure(output, target)
        if error < min_error:
            min_error = error
            offspring = tree_func(subtree, library_procedure)

    for unary_operator in unary_operators:
        transformed_subtree_semantics = pset.context[unary_operator.__name__](subtree_semantics)
        for residual_func, output_func, tree_func in binary_operations:
            residual = residual_func(transformed_subtree_semantics, target)
            library_procedure, procedure_semantics = library_selector(residual)
            output = output_func(transformed_subtree_semantics, procedure_semantics)
            error = distance_measure(output, target)
            if error < min_error:
                min_error = error
                offspring = tree_func([pset.mapping[unary_operator.__name__]] + subtree, library_procedure)

    if offspring is None:
        return ind,

    ind[:] = offspring
    return ind,
