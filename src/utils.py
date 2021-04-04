import bisect
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
        self.reverse_drone_can_serve()

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
        self.reverse_drone_can_serve()
        self.i_pot = self.data[0, 1:3]
        self.drone_distances = [distance.euclidean((self.data[i, 1:3]), self.i_pot)
                                if self.data[i, 3] == 1 else float('inf')
                                for i in range(len(self.data))]
        self.truck_distances = [[distance.cityblock(self.data[i, 1:3], self.data[j, 1:3])
                                 for i in range(len(self.data))] for j in range(len(self.data))]

    def reverse_drone_can_serve(self):
        for i in self.data:
            if i[3] == 0:
                i[3] = 1
            else:
                i[3] = 0

        self.data[0, 3] = 0

    def cal_time2serve_by_truck(self, individual: list, new_method=False):
        result = 0
        if not new_method:
            city_served_by_truck_list = [i for i, v in enumerate(individual) if v == 0]
            cost_matrix = np.array([[self.truck_distances[i][j]
                                     for i in city_served_by_truck_list] for j in city_served_by_truck_list])
            route_index = elkai.solve_float_matrix(cost_matrix, runs=1)
            result = sum([cost_matrix[i][i + 1] for i in range(-1, len(route_index) - 1)]) / self.truck_speed

        else:
            result = 0
            for i in individual:
                if individual[i] != -1:
                    if i + 1 not in individual:
                        result += self.truck_distances[0][i + 1]
                    else:
                        result += self.truck_distances[individual[i]][i + 1]
        return result

    def cal_time2serve_by_drones(self, individual: list, new_method=False):
        dist_list = [self.drone_distances[index] for index, value in enumerate(individual) if value != 0]
        if new_method:
            dist_list = [self.drone_distances[index + 1] for index, value in enumerate(individual) if value == -1]

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

    """
        new method
    """

    def init_individual_new_method(self):
        size = len(self.data) - 1

        ind = [-1] * size

        cus_can_served_by_drone = [i for i in range(1, size + 1) if self.data[i, 3] == 1]
        cus_served_by_drone = random.sample(cus_can_served_by_drone,
                                            k=random.randint(0, len(cus_can_served_by_drone)))
        # for i in cus_served_by_drone:
        #     ind[i - 1] = 0

        cus_served_by_truck = [i for i in range(1, size + 1) if i not in cus_served_by_drone]
        cur = 0
        while len(cus_served_by_truck) > 0:
            cus_served_by_truck.sort(key=lambda x: self.truck_distances[cur][x])

            ind[cus_served_by_truck[0] - 1] = cur
            cur = cus_served_by_truck[0]
            cus_served_by_truck.remove(cur)
        return ind

    @staticmethod
    def get_cus_served_by_drone(ind):
        return [i + 1 for i, v in enumerate(ind) if v == -1]

    @staticmethod
    def get_cus_served_by_truck(ind):
        result = []
        cur = 0
        while cur in ind:
            cur = ind.index(cur) + 1
            result.append(cur)

        return result

    def cal_fitness_new_method(self, individual):
        return max(self.cal_time2serve_by_truck(individual=individual, new_method=True),
                   self.cal_time2serve_by_drones(individual=individual, new_method=True))

    def crossover_new_method(self, ind1, ind2):
        cus_served_by_truck1 = self.get_cus_served_by_truck(ind1)
        cus_served_by_truck2 = self.get_cus_served_by_truck(ind2)

        same_truck_cus = set(cus_served_by_truck1).intersection(cus_served_by_truck2)
        same_truck_cus_i1 = []
        same_truck_cus_i2 = []
        for i in same_truck_cus:
            bisect.insort(same_truck_cus_i1, cus_served_by_truck1.index(i))
            bisect.insort(same_truck_cus_i2, cus_served_by_truck2.index(i))

        order1 = [cus_served_by_truck1[i] for i in same_truck_cus_i1]
        order2 = [cus_served_by_truck2[i] for i in same_truck_cus_i2]

        for i, v in enumerate(same_truck_cus_i1):
            cus_served_by_truck1[v] = order2[i]

        for i, v in enumerate(same_truck_cus_i2):
            cus_served_by_truck2[v] = order1[i]

        for i, v in enumerate(cus_served_by_truck1):
            if i == 0:
                ind1[v-1] = 0
            else:
                ind1[v-1] = cus_served_by_truck1[i-1]

        for i, v in enumerate(cus_served_by_truck2):
            if i == 0:
                ind2[v-1] = 0
            else:
                ind2[v-1] = cus_served_by_truck2[i-1]

        return ind1, ind2

    def mutate_new_method(self, ind, prob):
        cus_served_by_drone = self.get_cus_served_by_drone(ind)
        cus_served_by_truck = self.get_cus_served_by_truck(ind)

        if random.random() < prob:
            i1, i2 = random.sample(range(len(ind)), k=2)
            cus1, cus2 = i1 + 1, i2 + 1
            while (cus1 in cus_served_by_drone and cus2 in cus_served_by_drone) \
                    or (cus1 in cus_served_by_truck and cus2 in cus_served_by_drone and self.data[cus1, 3] == 0) \
                    or (cus2 in cus_served_by_truck and cus1 in cus_served_by_drone and self.data[cus2 + 1, 3] == 0):
                i1, i2 = random.sample(range(len(ind)), k=2)
                cus1, cus2 = i1 + 1, i2 + 1

            if cus2 in cus_served_by_drone:
                i1, i2 = i2, i1
                cus1, cus2 = i1 + 1, i2 + 1

            if cus1 in cus_served_by_truck and cus2 in cus_served_by_truck:
                if cus1 in ind:
                    ind[ind.index(cus1)] = cus2
                if i2 + 1 in ind:
                    ind[ind.index(cus2)] = cus1
            else:
                ind[ind.index(cus2)] = cus1

            ind[i1], ind[i2] = ind[i2], ind[i1]
        return ind,


if __name__ == '__main__':
    a = [-1, -1, 0, 5, 7, 4, 3]
    b = [5, 0, 1, -1, 2, -1, -1]
    Utils.get_instance().crossover_new_method(a, b)
