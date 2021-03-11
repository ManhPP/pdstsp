import glob
from itertools import combinations
from init_log import init_log
from arg_parser import parse_config
import pandas as pd
from scipy.spatial import distance
import elkai
import numpy as np
import random


def cx_random_respect(ind1, ind2):
    for i in range(0, len(ind1)):
        if ind1[i] != ind2[i]:
            if random.uniform(0, 1) < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


class Utils:
    __utils = None

    def __init__(self):
        constants, ga_config, self.data_path = parse_config()

        self.data_files = glob.glob(self.data_path)
        self.truck_speed = constants["truck_speed"]
        self.drone_speed = constants["drone_speed"]
        # self.speed = constants["speed"]
        self.num_drones = constants["num_drones"]
        self.data = pd.read_csv(self.data_files[0], header=None).to_numpy()[:-1]

        for i in self.data:
            if i[3] == 0:
                i[3] = 1
            else:
                i[3] = 0

        self.terminate = ga_config["terminate"]
        self.pop_size = ga_config["pop_size"]
        self.num_generation = ga_config["num_generation"]
        self.cx_pb = ga_config["cx_pb"]
        self.mut_pb = ga_config["mut_pb"]
        self.num_run = ga_config["num_run"]
        self.i_pot = self.data[0, 1:3]
        self.drone_distances = [distance.euclidean((self.data[i, 1:3]), self.i_pot)
                                if self.data[i, 3] == 1 else float('inf')
                                for i in range(len(self.data))]
        self.truck_distances = [[distance.cityblock(self.data[i, 1:3], self.data[j, 1:3])
                                 for i in range(len(self.data))] for j in range(len(self.data))]

    @classmethod
    def get_instance(cls):
        if cls.__utils is None:
            cls.__utils = Utils()
        return cls.__utils

    def change_data(self, path):
        self.data = pd.read_csv(path, header=None).to_numpy()[:-1]

    def cal_time2serve_by_truck(self, individual: list):
        city_served_by_truck_list = [i for i, v in enumerate(individual) if v == 0]
        cost_matrix = np.array([[self.truck_distances[i][j]
                                 for i in city_served_by_truck_list] for j in city_served_by_truck_list])
        route_index = elkai.solve_float_matrix(cost_matrix, runs=1)

        return sum([cost_matrix[i][i + 1] for i in range(-1, len(route_index) - 1)]) / self.truck_speed

    def cal_time2serve_by_drones(self, individual: list):
        dist_list = [self.drone_distances[index] for index, value in enumerate(individual) if value != 0]

        if self.num_drones == 1:
            return 2 / self.drone_speed * sum(dist_list)

        dist_list.sort()
        if len(dist_list) < self.num_drones:
            return 2 * dist_list.pop() / self.drone_speed

        drones = np.zeros(self.num_drones)
        for i in range(self.num_drones):
            drones[i] = dist_list.pop()
        dist = 0
        while len(dist_list) > 0:
            drones.sort()
            dist += drones[0]
            drones -= drones[0]
            drones[0] = dist_list.pop()

        dist += max(drones)
        return 2 * dist / self.drone_speed

    def cal_fitness(self, individual: list):
        return max(self.cal_time2serve_by_truck(individual=individual),
                   self.cal_time2serve_by_drones(individual=individual))

    def init_individual(self, size):
        ind = [random.randint(0, 1) if self.data[i, 3] == 1 else 0 for i in range(size)]
        ind[0] = 0
        return ind

    def mutate_flip_bit(self, individual, ind_pb):
        for i in range(1, len(individual)):
            if random.random() < ind_pb and (self.data[i, 3] == 1 or individual[i] == 1):
                individual[i] = type(individual[i])(not individual[i])
        return individual,

    def cal_drone_time_matrix(self):
        return [2 / self.drone_speed * self.drone_distances[i]
                for i in range(len(self.data))]

    def cal_truck_time_matrix(self):
        return [[self.truck_distances[i][j] / self.truck_speed
                 for i in range(len(self.data))] for j in range(len(self.data))]

    def get_nodes_can_served_by_drone(self):
        return [i for i in range(1, len(self.data)) if self.data[i, 3] == 1]

    def get_sub_node_lists(self):
        for i in range(1, len(self.data) + 1):
            for j in combinations(range(len(self.data)), i):
                if 0 in j:
                    yield j


if __name__ == '__main__':
    # print(Utils.get_instance().cal_fitness([0, 0, 1]))
    # var = Utils.get_instance()
    # for _ in range(10):
    #     print(Utils.get_instance().init_individual(5))
    logger = init_log()
    result = []
    logger.info("runs = 10 / runs = 1: ")
    for i in range(15):
        ind = Utils.get_instance().init_individual(len(Utils.get_instance().data))
        comp = Utils.get_instance().cal_time2serve_by_truck(ind)
        logger.info("No " + str(i) + ": " + str(comp))
        result.append(comp)

    avg = np.mean(result)
    std = np.std(result)
    mi = np.min(result)
    ma = np.max(result)
    logger.info([mi, ma, avg, std])
