from ortools.linear_solver import pywraplp

from init_log import init_log
from utils import Utils


def solve(logger):
    if logger is None:
        raise Exception("Error: logger is None!")

    solver = pywraplp.Solver('wsn',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    num_all_node = len(Utils.get_instance().data)
    num_drone = Utils.get_instance().num_drones

    c_u = Utils.get_instance().get_nodes_can_served_by_drone()  # phuc vu duoc boi drone

    x = {}
    y = {}

    alpha = solver.NumVar(0, solver.infinity(), "time")

    for i in range(num_all_node):
        for j in range(num_all_node):
            x[i, j] = solver.BoolVar("x[%i, %i]" % (i, j))

    for i in range(num_all_node):
        for k in range(num_drone):
            y[i, k] = solver.BoolVar("y[%i, %i]" % (i, k))

    solver.Minimize(alpha)

    drone_time_matrix = Utils.get_instance().cal_drone_time_matrix()
    truck_time_matrix = Utils.get_instance().cal_truck_time_matrix()

    # 2
    solver.Add(alpha >= solver.Sum(truck_time_matrix[i][j] * x[i, j]
                                   for i in range(num_all_node) for j in range(num_all_node)))
    # 3
    for k in range(num_drone):
        solver.Add(alpha >= solver.Sum(drone_time_matrix[i] * y[i, k] for i in c_u))

    # 4
    # voi node co the phuc vu boi drone
    for j in c_u:
        solver.Add(solver.Sum(x[i, j] for i in range(num_all_node))
                   + solver.Sum(y[j, k] for k in range(num_drone)) == 1)

    # voi node khong the phuc vu boi drone
    for j in range(1, num_all_node):
        if j not in c_u:
            solver.Add(solver.Sum(x[i, j] for i in range(num_all_node)) == 1)

    # 5
    for i in range(1, num_all_node):
        if i not in c_u:
            solver.Add(solver.Sum(x[i, j] for j in range(num_all_node)) == 1)

    for i in c_u:
        solver.Add(solver.Sum(x[i, j] for j in range(num_all_node))
                   + solver.Sum(y[i, k] for k in range(num_drone)) == 1)

    # 6
    for i in range(num_all_node):
        solver.Add(solver.Sum(x[j, i] for j in range(num_all_node))
                   == solver.Sum(x[i, h] for h in range(num_all_node)))

    # 7
    for s in Utils.get_instance().get_sub_node_lists():
        solver.Add(solver.Sum(x[i, j] for i in s for j in range(num_all_node) if j not in s)
                   + solver.Sum(y[i, k] for i in s for k in range(num_drone)) >= 1)

    print('Number of constraints =', solver.NumConstraints())
    result_status = solver.Solve()
    assert result_status == pywraplp.Solver.OPTIMAL
    print('optimal value = ', solver.Objective().Value())
    print()
    print("Time = ", solver.WallTime(), " milliseconds")


if __name__ == '__main__':
    log = init_log()
    log.info("Start running IP...")

    for path in Utils.get_instance().data_files:
        Utils.get_instance().load_data(path)
        log.info("input path: %s" % path)
        solve(logger=log)
