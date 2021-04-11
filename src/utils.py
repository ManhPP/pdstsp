import bisect
import glob
import random
from itertools import combinations

import elkai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance

from arg_parser import parse_config


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

        self.terminate = ga_config["terminate"]
        self.pop_size = ga_config["pop_size"]
        self.num_generation = ga_config["num_generation"]
        self.cx_pb = ga_config["cx_pb"]
        self.mut_pb = ga_config["mut_pb"]
        self.num_run = ga_config["num_run"]

        self.data = None
        self.i_pot = None
        self.cus_can_served_by_drone = None
        self.drone_distances = None
        self.truck_distances = None

        self.load_data(self.data_files[0])

    @classmethod
    def get_instance(cls):
        if cls.__utils is None:
            cls.__utils = Utils()
        return cls.__utils

    def load_data(self, path):
        self.data = pd.read_csv(path, header=None).to_numpy()[:-1]
        self.reverse_drone_can_serve()
        self.i_pot = self.data[0, 1:3]
        self.cus_can_served_by_drone = [i for i in range(len(self.data)) if self.data[i, 3] == 1]
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

    def cal_time2serve_by_truck(self, individual: list, new_method=True):
        result = 0
        if not new_method:
            city_served_by_truck_list = [i for i, v in enumerate(individual) if v == 0]
            cost_matrix = np.array([[self.truck_distances[i][j]
                                     for i in city_served_by_truck_list] for j in city_served_by_truck_list])
            route_index = elkai.solve_float_matrix(cost_matrix, runs=1)
            result = sum([cost_matrix[i][i + 1] for i in range(-1, len(route_index) - 1)])

        else:
            for i, v in enumerate(individual):
                if individual[i] != -1:
                    if i + 1 not in individual:
                        result += self.truck_distances[0][i + 1]
                    else:
                        result += self.truck_distances[v][i + 1]

        return result / self.truck_speed

    def cal_time2serve_by_drones(self, individual: list, new_method=True):
        dist_list = [self.drone_distances[index] for index, value in enumerate(individual) if value != 0]
        if new_method:
            dist_list = [self.drone_distances[index + 1] for index, value in enumerate(individual) if value == -1]

        if len(dist_list) == 0:
            return 0

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

    def cal_fitness(self, individual: list, new_method=True):
        return max(self.cal_time2serve_by_truck(individual=individual, new_method=new_method),
                   self.cal_time2serve_by_drones(individual=individual, new_method=new_method))

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
        # tmp = list(filter(lambda a: a != -1, ind))
        result = []
        cur = 0
        while cur in ind:
            cur = ind.index(cur) + 1
            result.append(cur)

        return result

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
                ind1[v - 1] = 0
            else:
                ind1[v - 1] = cus_served_by_truck1[i - 1]

        for i, v in enumerate(cus_served_by_truck2):
            if i == 0:
                ind2[v - 1] = 0
            else:
                ind2[v - 1] = cus_served_by_truck2[i - 1]

        return ind1, ind2

    def mutate_new_method(self, ind, prob=0.5):
        cus_served_by_drone = self.get_cus_served_by_drone(ind)
        cus_served_by_truck = self.get_cus_served_by_truck(ind)
        rate = random.random()
        if len(self.cus_can_served_by_drone) == 0:
            rate = -1
        if float('inf') not in self.drone_distances[1:]:
            rate = 2

        if rate < prob:
            cus1, cus2 = random.sample(range(1, len(ind)+1), k=2)
            while (cus1 in cus_served_by_drone and cus2 in cus_served_by_drone) \
                    or (cus1 in cus_served_by_truck and cus2 in cus_served_by_drone and self.data[cus1, 3] == 0) \
                    or (cus2 in cus_served_by_truck and cus1 in cus_served_by_drone and self.data[cus2, 3] == 0):
                cus1, cus2 = random.sample(range(1, len(ind) + 1), k=2)

            if cus2 in cus_served_by_drone:
                cus1, cus2 = cus2, cus1

            if cus1 in cus_served_by_truck and cus2 in cus_served_by_truck:
                i1, i2 = cus_served_by_truck.index(cus1), cus_served_by_truck.index(cus2)
                cus_served_by_truck[i1], cus_served_by_truck[i2] = cus_served_by_truck[i2], cus_served_by_truck[i1]
            else:
                i2 = cus_served_by_truck.index(cus2)
                cus_served_by_truck[i2] = cus1
                ind[cus2 - 1] = -1

            for i, v in enumerate(cus_served_by_truck):
                if i == 0:
                    ind[v - 1] = 0
                else:
                    ind[v - 1] = cus_served_by_truck[i - 1]

        else:
            time2serve_by_truck = self.cal_time2serve_by_truck(individual=ind, new_method=True)
            time2serve_by_drones = self.cal_time2serve_by_drones(individual=ind, new_method=True)

            if time2serve_by_drones > time2serve_by_truck:
                cus = random.choice(cus_served_by_drone)
            elif time2serve_by_drones < time2serve_by_truck:
                cus = random.choice(cus_served_by_truck)
            else:
                cus = random.choice(self.cus_can_served_by_drone)

            if ind[cus - 1] == -1:
                cus_served_by_truck.insert(random.randint(0, len(cus_served_by_truck)), cus)
                for i, v in enumerate(cus_served_by_truck):
                    if i == 0:
                        ind[v - 1] = 0
                    else:
                        ind[v - 1] = cus_served_by_truck[i - 1]

            else:
                if cus in ind:
                    ind[ind.index(cus)] = ind[cus-1]
                ind[cus - 1] = -1

        return ind,


if __name__ == '__main__':
    a = [-1, -1, 0, 5, 7, 4, 3]
    b = [5, 0, 1, -1, 2, -1, -1]

    Utils.get_instance().mutate_new_method(a, 1)

    # d = [2, 21, 1, 3, 4, 9, 5, 7, 8, 11, 52, 16, 12, 13, 14, 20, 10, 17, 18, 19, 22, 15, 24, 25, 85, -1, -1, 6, -1,
    # -1, -1, 40, -1, -1, -1, 28, -1, -1, -1, 43, -1, -1, 45, 213, 44, -1, 48, 49, 188, 184, 53, 51, 63, 55, 56, 57,
    # 83, 54, 58, 59, 60, 61, 66, 62, 64, 65, 68, 69, 117, 71, 0, 70, 72, 75, 76, 77, 78, 73, 88, 87, 82, 84, 79, 23,
    # 86, 89, 81, 80, 90, 91, 74, 36, -1, -1, -1, 104, -1, -1, -1, 92, 100, 101, 102, 32, -1, -1, -1, 113, -1, -1,
    # -1, 103, 96, -1, 112, 120, 116, 115, 118, 123, 119, 121, 122, 108, -1, -1, -1, 124, 128, 129, 130, 131, -1,
    # 136, 134, 137, 138, 132, 142, 141, 152, 140, 139, 143, 144, 145, 146, 147, -1, 204, 155, 153, 154, 151, 150,
    # 159, 156, 157, 160, 161, 162, 199, 172, 163, 164, -1, -1, 176, -1, 148, -1, 135, 165, -1, -1, 173, -1, -1, -1,
    # 168, -1, -1, -1, 47, -1, 180, 186, 189, 190, 191, 193, 194, 192, 195, 196, 198, 187, 197, 50, 223, 200, 203,
    # 211, 158, 170, 205, 206, 207, 208, 209, 210, 201, 214, 215, 216, 219, 218, 227, 220, 212, 202, 221, 222, 225,
    # 217, 228, 226, 229, 67]
    #
    # G = nx.Graph()
    #
    # for i, v in enumerate(d):
    #     p = v
    #     if v == -1:
    #         continue
    #
    #     G.add_edge(p, i + 1)
    # pos = nx.spring_layout(G, scale=5)
    # nx.draw_networkx(G, pos)

    plt.show()
