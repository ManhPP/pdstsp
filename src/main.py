import numpy as np
from deap import base, creator, tools, algorithms
from init_log import init_log
from utils import Utils
from utils import cx_random_respect

creator.create("FitnessMin", base.Fitness, weights=(-1.,))
FitnessMin = creator.FitnessMin
creator.create("Individual", list, fitness=FitnessMin)


def init_individual():
    return creator.Individual(Utils.get_instance().init_individual_new_method())


def run_ga(logger):
    if logger is None:
        raise Exception("Error: logger is None!")

    stats = tools.Statistics(lambda individual: individual.fitness.values)
    stats.register("best", np.min, axis=0)

    logbook = tools.Logbook()
    logbook.header = 'ga', "best"
    result = []

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", Utils.get_instance().crossover_new_method)
    toolbox.register("mutate", Utils.get_instance().mutate_new_method)
    toolbox.register("select", tools.selTournament, tournsize=10)
    toolbox.register("evaluate", Utils.get_instance().cal_fitness)

    for time in range(Utils.get_instance().num_run):
        pop = toolbox.population(Utils.get_instance().pop_size)
        best_ind = toolbox.clone(pop[0])

        prev = -1  # use for termination
        count_term = 0  # use for termination

        for _ in range(Utils.get_instance().num_generation):
            offsprings = map(toolbox.clone, toolbox.select(pop, len(pop) - 1))
            offsprings = algorithms.varAnd(offsprings, toolbox, Utils.get_instance().cx_pb, Utils.get_instance().mut_pb)
            offsprings.append(best_ind)

            min_value = float('inf')
            invalid_individuals = []

            fitness = toolbox.map(toolbox.evaluate, offsprings)
            for ind, fit in zip(offsprings, fitness):
                if fit == float('inf'):
                    print("!!!")
                else:
                    invalid_individuals.append(ind)

            fitness = toolbox.map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitness):
                ind.fitness.values = [fit]
                if min_value > fit:
                    min_value = fit
                    best_ind = toolbox.clone(ind)

            b = round(min_value, 6)
            if prev == b:
                count_term += 1
            else:
                count_term = 0

            pop[:] = invalid_individuals[:]
            logger.info(f"{time} - {_} : {b}")
            prev = b
            if count_term == Utils.get_instance().terminate:
                break

        record = stats.compile(pop)
        logbook.record(ga=time + 1, **record)
        logger.info(logbook.stream)
        result.append(min_value)

    avg = np.mean(result)
    std = np.std(result)
    mi = np.min(result)
    ma = np.max(result)
    logger.info([mi, ma, avg, std])


if __name__ == '__main__':
    log = init_log()
    log.info("Start running GA...")

    for path in Utils.get_instance().data_files:
        Utils.get_instance().load_data(path)
        log.info("input path: %s" % path)
        run_ga(logger=log)
