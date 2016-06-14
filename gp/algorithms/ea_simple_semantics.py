import random
import time

from deap import tools


def assign_raw_fitness(population):
    for ind in population:
        ind.fitness.values = (ind.error,)


def breed(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def ea_simple(population, toolbox, cxpb, mutpb, ngen, elite_size=0, stats=None, archive=None, verbose=False):
    start = time.time()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'cpu_time'] + (stats.fields if stats else [])

    for ind in population:
        ind.error = toolbox.evaluate_error(ind)[0]
    toolbox.assign_fitness(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), cpu_time=time.time() - start, **record)

    if archive is not None:
        archive.update(population)
    if verbose:
        print logbook.stream

    for gen in range(1, ngen + 1):
        elite = tools.selBest(population, elite_size)
        elite = [toolbox.clone(ind) for ind in elite]

        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = breed(offspring, toolbox, cxpb, mutpb)
        offspring = offspring + elite

        for ind in offspring:
            ind.error = toolbox.evaluate_error(ind)[0]
        toolbox.assign_fitness(offspring)

        if hasattr(toolbox, "simplify"):
            toolbox.simplify(offspring)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), cpu_time=time.time() - start, **record)
        if archive is not None:
            archive.update(population)
        if verbose:
            print logbook.stream

    return population, logbook
