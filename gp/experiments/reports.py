import csv
import operator

import numpy
from deap import tools


def configure_inf_protected_stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("avg", lambda values: numpy.mean(filter(numpy.isfinite, values)))
    mstats.register("std", lambda values: numpy.std(filter(numpy.isfinite, values)))
    mstats.register("min", lambda values: numpy.min(filter(numpy.isfinite, values)))
    mstats.register("max", lambda values: numpy.max(filter(numpy.isfinite, values)))

    stats_best_ind = tools.Statistics(lambda ind: (ind.fitness.values[0], len(ind)))
    stats_best_ind.register("size_min", lambda values: min(values)[1])
    stats_best_ind.register("size_max", lambda values: max(values)[1])
    mstats["best_tree"] = stats_best_ind
    return mstats


def save_log_to_csv(pop, log, file_name):
    columns = [log.select("cpu_time")]
    columns_names = ["cpu_time"]
    for chapter_name, chapter in log.chapters.items():
        for column in chapter[0].keys():
            columns_names.append(str(column) + "_" + str(chapter_name))
            columns.append(chapter.select(column))

    rows = zip(*columns)
    with open(file_name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(columns_names)
        for row in rows:
            writer.writerow(row)


def save_archive(archive):
    def decorator(func):
        def wrapper(pop, log, file_name):
            func(pop, log, file_name)
            archive.save(file_name)

        return wrapper

    return decorator

