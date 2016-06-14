import csv
import operator
from collections import defaultdict
from copy import deepcopy

import numpy
from deap.tools import HallOfFame

from gp.semantic import semantics, distances


class HistoricalHallOfFame(HallOfFame):
    def __init__(self, maxsize, similar=operator.eq):
        super(HistoricalHallOfFame, self).__init__(maxsize, similar)
        self.historical_trees = list()

    def update(self, population):
        super(HistoricalHallOfFame, self).update(population)
        best_tree = max(population, key=operator.attrgetter("fitness"))
        self.historical_trees.append(deepcopy(best_tree))


class BestTreeArchive(object):
    def __init__(self, frequency):
        self.frequency = frequency
        self.generation_counter = 0
        self.generations = []
        self.best_trees = []

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            best_tree = min(population, key=operator.attrgetter("error"))
            self.best_trees.append(deepcopy(best_tree))
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        best_tree_file = "best_tree_" + log_file
        with open(best_tree_file, 'wb') as f:
            writer = csv.writer(f)
            for gen, best_tree in zip(self.generations, self.best_trees):
                writer.writerow([gen, best_tree])


class MultiArchive(object):
    def __init__(self, archives):
        self.archives = archives

    def update(self, population):
        for archive in self.archives:
            archive.update(population)

    def save(self, log_file):
        for archive in self.archives:
            archive.save(log_file)


class TestSetPerformanceArchive(object):
    def __init__(self, frequency, evaluate_test_error):
        self.evaluate_test_error = evaluate_test_error
        self.test_errors = []
        self.frequency = frequency
        self.generation_counter = 0
        self.generations = []

    def update(self, population):
        if self.generation_counter % self.frequency == 0:
            best_tree = min(population, key=operator.attrgetter("error"))
            test_error = self.evaluate_test_error(best_tree)[0]
            self.test_errors.append(test_error)
            self.generations.append(self.generation_counter)
        self.generation_counter += 1

    def save(self, log_file):
        test_error_file = "test_error_" + log_file
        with open(test_error_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "test_error"])
            for gen, test_error in zip(self.generations, self.test_errors):
                writer.writerow([gen, test_error])


class TreeCountingArchive(object):
    def __init__(self, file_prefix="tree_num_", ind_selector=None):
        self.ind_selector = ind_selector
        self.num_distinct_trees = []
        self.num_sem_distinct_trees = []
        self.num_sem_distinct_trees_threshold = []
        self.num_infinite_trees = []
        self.num_constant_trees = []
        self.num_novel_trees = []
        self.syntactic_entropy = []
        self.semantic_entropy = []
        self.fitness_entropy = []
        self.gini_index = []
        self.file_prefix = file_prefix

    def update(self, population):
        num_infinite_trees = 0
        num_constant_trees = 0

        fitness_count_dict = defaultdict(int)
        tree_expression_dict = defaultdict(int)
        tree_semantics_set = set()
        tree_semantics_list = list()

        inds = population if self.ind_selector is None else self.ind_selector(population)
        for ind in inds:
            tree_semantics = ind.semantics[0]
            tree_expression_dict[tree_semantics.expr] += 1
            tree_semantics_set.add(tree_semantics.tostring())
            tree_semantics_list.append(tree_semantics)
            fitness_count_dict[ind.error] += 1

            if not numpy.isfinite(numpy.sum(tree_semantics)):
                num_infinite_trees += 1
            elif numpy.ptp(tree_semantics) <= 10e-8:
                num_constant_trees += 1

        unique_semantics, semantic_groups = semantics.get_unique_semantics(tree_semantics_list, 10e-6,
                                                                           distances.cumulative_absolute_difference)
        self.semantic_entropy.append(calculate_entropy(semantic_groups))
        self.syntactic_entropy.append(calculate_entropy(tree_expression_dict.values()))
        self.fitness_entropy.append(calculate_entropy(fitness_count_dict.values()))
        self.gini_index.append(calculate_gini_index([ind.error for ind in population]))

        self.num_infinite_trees.append(num_infinite_trees)
        self.num_constant_trees.append(num_constant_trees)
        self.num_distinct_trees.append(len(tree_expression_dict))
        self.num_sem_distinct_trees.append(len(tree_semantics_set))
        self.num_sem_distinct_trees_threshold.append(len(unique_semantics))

    def save(self, log_file):
        semantic_statistics_file = self.file_prefix + log_file
        with open(semantic_statistics_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["gen",
                 "num_infinite_trees",
                 "num_constant_trees",
                 "num_distinct_trees",
                 "num_sem_distinct_trees",
                 "num_sem_distinct_trees_threshold",
                 "syntactic_entropy",
                 "semantic_entropy",
                 "fitness_entropy",
                 "gini_index"])
            for gen in range(len(self.num_infinite_trees)):
                writer.writerow(
                    [gen,
                     self.num_infinite_trees[gen],
                     self.num_constant_trees[gen],
                     self.num_distinct_trees[gen],
                     self.num_sem_distinct_trees[gen],
                     self.num_sem_distinct_trees_threshold[gen],
                     self.syntactic_entropy[gen],
                     self.semantic_entropy[gen],
                     self.fitness_entropy[gen],
                     self.gini_index[gen]])


def calculate_entropy(values):
    n = float(sum(values))
    if n == 0:
        return 0

    entropy = 0
    for v in values:
        entropy -= v * numpy.log2(v)
    return entropy / n + numpy.log2(n)


def calculate_gini_index(values):
    md = 0
    n = len(values)
    for i in range(n):
        for j in range(n):
            md += abs(values[i] - values[j])
    md /= n * n
    rmd = md / numpy.mean(values)
    gini = rmd / 2.0
    return gini
