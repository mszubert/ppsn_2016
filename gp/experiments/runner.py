import logging
import random

import numpy

import benchmark_problems


def configure_basic_logging(filename, level=logging.DEBUG):
    logging.basicConfig(filename=filename, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=level)


def configure_advanced_logging(filename, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)

    if len(logger.handlers) > 0:
        logger.handlers[0].stream.close()
        logger.removeHandler(logger.handlers[0])

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def run(random_seed, predictors, response, toolbox_functions, algorithm_names, benchmark_name="",
        test_predictors=None, test_response=None):
    """Runs sequentially given regression algorithms on a given dataset

    :param random_seed: seed for the random generator
    :param predictors: a list of tuples, each of which specifies values of predictor variables for a single observation
    :param response: a list of observed values of response variable (corresponding to observations in predictors list)
    :param toolbox_functions: a list of functions each of which produces a configuration of a single learning algorithm
    :param benchmark_name: optional name of the problem being solved, to be included in a produced csv result file
    """

    for toolbox_func, algorithm_name in zip(toolbox_functions, algorithm_names):
        if test_predictors is None or test_response is None:
            toolbox = toolbox_func(predictors=predictors, response=response)
        else:
            toolbox = toolbox_func(predictors=predictors, response=response, test_predictors=test_predictors,
                                   test_response=test_response)

        logging.info("Starting algorithm %s", algorithm_name)
        pop, log = toolbox.run()

        logging.info("Saving results of algorithm %s", algorithm_name)
        log_file_name = "{}_{}_{}.log".format(algorithm_name, benchmark_name, random_seed)
        toolbox.save(pop, log, log_file_name)


def run_benchmarks(random_seed, benchmarks, toolbox_functions, algorithm_names, logging_level=logging.INFO,
                   num_points=20, sample_evenly=False):
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    configure_advanced_logging("debug_{}.log".format(random_seed), level=logging_level)
    for benchmark in benchmarks:
        logging.info("Starting benchmark %s", benchmark.func_name)
        predictors, response = benchmark_problems.get_training_set(benchmark, num_points=num_points,
                                                                   sample_evenly=sample_evenly)
        run(random_seed, predictors, response, toolbox_functions, algorithm_names, benchmark.func_name)


def run_benchmarks_train_test(random_seed, benchmarks, toolbox_functions, algorithm_names, training_set_generator,
                              test_set_generator=None, logging_level=logging.INFO):
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    configure_advanced_logging("debug_{}.log".format(random_seed), level=logging_level)
    for benchmark in benchmarks:
        logging.info("Starting benchmark %s", benchmark.func_name)
        predictors, response = training_set_generator(benchmark)
        if test_set_generator is not None:
            test_predictors, test_response = test_set_generator(benchmark)
            run(random_seed, predictors, response, toolbox_functions, algorithm_names, benchmark.func_name,
                test_predictors=test_predictors, test_response=test_response)
        else:
            run(random_seed, predictors, response, toolbox_functions, algorithm_names, benchmark.func_name)
