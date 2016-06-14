from functools import partial
import logging
import operator
import sys

import cachetools

from deap import creator, base, tools, gp
import numpy

from gp.algorithms import ea_simple_semantics, initialization, operators, archive
from gp.experiments import symbreg, reports, fast_evaluate, runner, benchmark_problems
from gp.semantic import library, distances, semantics, fpm, backpropagation

NGEN = 100
POP_SIZE = 256
TOURNAMENT_SIZE = 7
MIN_DEPTH_INIT = 2
MAX_DEPTH_INIT = 6
MAX_HEIGHT = 17
MAX_SIZE = 300
LIBRARY_SEARCH_NEIGHBORS = 1
LIBRARY_CONST_REPLACEMENT_PROB = 1.0

ERROR_FUNCTION = fast_evaluate.root_mean_square_error


def get_archive(pset, test_predictors, test_response):
    tree_counting_archive = archive.TreeCountingArchive()

    test_expression_dict = cachetools.LRUCache(maxsize=1000)
    test_error_func = partial(ERROR_FUNCTION, response=test_response)
    evaluate_test_error = partial(fast_evaluate.fast_numpy_evaluate, context=pset.context, predictors=test_predictors,
                                  error_function=test_error_func, expression_dict=test_expression_dict)
    test_set_error_archive = archive.TestSetPerformanceArchive(frequency=1, evaluate_test_error=evaluate_test_error)

    best_tree_archive = archive.BestTreeArchive(frequency=10)
    multi_archive = archive.MultiArchive([tree_counting_archive, test_set_error_archive, best_tree_archive])
    return multi_archive


def get_generic_toolbox(predictors, response, test_predictors, test_response, register_mut_func, mut_pb, xover_pb,
                        node_selector, size_limit=MAX_SIZE):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    pset = symbreg.get_numpy_pset(len(predictors[0]))
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_DEPTH_INIT, max_=MAX_DEPTH_INIT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", initialization.syntactically_distinct, individual=toolbox.individual, retries=100)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    expression_dict = cachetools.LRUCache(maxsize=10000)
    toolbox.register("semantic_calculator", semantics.calculate_semantics, context=pset.context, predictors=predictors,
                     expression_dict=expression_dict)

    toolbox.register("error_func", ERROR_FUNCTION, response=response)
    toolbox.register("evaluate_error", semantics.calc_eval_semantics, context=pset.context, predictors=predictors,
                     eval_semantics=toolbox.error_func, expression_dict=expression_dict)
    toolbox.register("assign_fitness", ea_simple_semantics.assign_raw_fitness)

    toolbox.register("mate", operators.one_point_xover_biased, node_selector=node_selector)
    toolbox.decorate("mate", operators.static_limit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", operators.static_limit(key=len, max_value=size_limit))

    lib = library.PopulationLibrary()
    register_mut_func(toolbox, lib, pset, response, node_selector)
    toolbox.decorate("mutate", operators.static_limit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mutate", operators.static_limit(key=len, max_value=size_limit))

    mstats = reports.configure_inf_protected_stats()
    multi_archive = get_archive(pset, test_predictors, test_response)
    multi_archive.archives.append(lib)

    pop = toolbox.population(n=POP_SIZE)
    toolbox.register("run", ea_simple_semantics.ea_simple, population=pop, toolbox=toolbox, cxpb=xover_pb,
                     mutpb=mut_pb, ngen=NGEN, elite_size=0, stats=mstats, verbose=True, archive=multi_archive)

    toolbox.register("save", reports.save_log_to_csv)
    toolbox.decorate("save", reports.save_archive(multi_archive))

    return toolbox


def register_srm(toolbox, lib, pset, response, node_selector):
    toolbox.register("grow", gp.genGrow, pset=pset, min_=1, max_=6)
    toolbox.register("mutate", operators.mutation_biased, expr=toolbox.grow,
                     node_selector=node_selector)


def register_rdo(toolbox, lib, pset, response, node_selector):
    toolbox.register("lib_selector", backpropagation.library_search, lib=lib,
                     distance_measure=distances.cumulative_absolute_difference, check_constant_prob=1.0)
    toolbox.register("mutate", backpropagation.rdo, library_selector=toolbox.lib_selector, target=response,
                     node_selector=node_selector, semantic_calculator=toolbox.semantic_calculator,
                     max_height=MAX_HEIGHT, inversion_map=backpropagation.get_numpy_inversion_map())


def register_fpm(toolbox, lib, pset, response, node_selector):
    toolbox.register("lib_selector", lib.get_closest_direction_scaled,
                     distance_measure=distances.cumulative_absolute_difference, pset=pset,
                     k=LIBRARY_SEARCH_NEIGHBORS, check_constant_prob=1.0, max_height=MAX_HEIGHT - 2)
    toolbox.register("mutate", fpm.forward_propagation_mutation_all, pset=pset,
                     library_selector=toolbox.lib_selector, semantic_calculator=toolbox.semantic_calculator,
                     distance_measure=distances.root_mean_square_error,
                     target=response, binary_operations=fpm.get_all_numpy_operations(pset),
                     unary_operators=[numpy.sin, numpy.cos, numpy.exp, symbreg.numpy_protected_log_abs],
                     node_selector=node_selector)


uniform_node_selector = operators.uniform_depth_node_selector

variants = []
variants_names = []
for mut_operator, mut_name in zip([register_srm, register_rdo, register_fpm], ["SRM", "RDO", "FPM"]):
    for xover_prob in [0.0, 0.5, 1.0]:
        for mut_prob in [0.5, 1.0]:
            variants.append(partial(get_generic_toolbox, register_mut_func=mut_operator, xover_pb=xover_prob,
                                    mut_pb=mut_prob, node_selector=uniform_node_selector, size_limit=1000000))
            variants_names.append("_".join([mut_name, str(xover_prob), str(mut_prob)]))

random_seed = int(sys.argv[1])
training_set_generator = partial(benchmark_problems.get_training_set, num_points=20, sample_evenly=True)
test_set_generator = partial(benchmark_problems.get_training_set, num_points=1000, sample_evenly=False)
runner.run_benchmarks_train_test(random_seed,
                                 [benchmark_problems.quartic, benchmark_problems.septic, benchmark_problems.nonic,
                                  benchmark_problems.r1, benchmark_problems.r2, benchmark_problems.r3],
                                 variants, variants_names,
                                 training_set_generator=training_set_generator,
                                 test_set_generator=test_set_generator,
                                 logging_level=logging.INFO)

training_set_generator = partial(benchmark_problems.get_training_set, num_points=100, sample_evenly=True)
runner.run_benchmarks_train_test(random_seed,
                                 [benchmark_problems.keijzer11, benchmark_problems.keijzer12,
                                  benchmark_problems.keijzer13, benchmark_problems.keijzer14,
                                  benchmark_problems.keijzer15],
                                 variants, variants_names,
                                 training_set_generator=training_set_generator,
                                 test_set_generator=test_set_generator,
                                 logging_level=logging.INFO)