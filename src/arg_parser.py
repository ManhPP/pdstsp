from argparse import ArgumentParser
import configparser
from pathlib import Path


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--pop-size', default=50, type=int)
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--cross-rate', default=0.7, type=float)
    parser.add_argument('--mut-rate', default=0.1, type=float)

    return parser.parse_args()


def parse_config():
    parser = configparser.ConfigParser()
    config_file_path = Path(__file__).parent.parent
    parser.read(f"{config_file_path}/config/config.ini")
    dict_constant = dict()
    dict_constant["truck_speed"] = parser.getfloat("CONSTANT", "truck_speed")
    dict_constant["drone_speed"] = parser.getfloat("CONSTANT", "drone_speed")
    # dict_constant["speed"] = parser.getfloat("CONSTANT", "speed")
    dict_constant["num_drones"] = parser.getint("CONSTANT", "num_drones")
    data_path = f"{config_file_path}/{parser.get('PATH', 'data_path')}"
    dict_ga_config = dict()
    dict_ga_config["terminate"] = parser.getint("GA", "terminate")
    dict_ga_config["pop_size"] = parser.getint("GA", "pop_size")
    dict_ga_config["num_generation"] = parser.getint("GA", "num_generation")
    dict_ga_config["cx_pb"] = parser.getfloat("GA", "cx_pb")
    dict_ga_config["mut_pb"] = parser.getfloat("GA", "mut_pb")
    dict_ga_config["num_run"] = parser.getint("GA", "num_run")
    return dict_constant, dict_ga_config, data_path


if __name__ == '__main__':
    # parse_config()
    # logger = init_log()
    # logger.info("arg_parser")
    # logger.warning("this is test")
    dict_const = parse_config()
    print(dict_const)
    # logger.info(dict_const)
