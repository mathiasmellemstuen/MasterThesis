import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
from config_parser import parse_config_file
from utils.get_device import get_device
from utils.split_only_last import split_only_last

if __name__ == "__main__":
    
    # Setting up and getting console arguments 
    argument_parser = argparse.ArgumentParser(prog="Learning by ranking")

    argument_parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    argument_parser.add_argument("-e", "--epochs", type=int)
    argument_parser.add_argument("-ex", "--experiment", type=int)
    argument_parser.add_argument("-si", "--save_interval", type=int)
    argument_parser.add_argument("-lr", "--learning_rate", type=float)
    argument_parser.add_argument("-b", "--batch_size", type=int)
    argument_parser.add_argument("-s", "--seed", type=int)
    argument_parser.add_argument("-c", "--config", type=str, default="configs/experiment_1.json")
    argument_parser.add_argument("-dw", "--dataloader_workers", type=int)
    argument_parser.add_argument("-dldl", "--dataloader_drop_last", action=argparse.BooleanOptionalAction, default=False)

    arguments = argument_parser.parse_args()
    arguments = parse_config_file(arguments)

    # Setting seed
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    print(f"Starting experiment {arguments.experiment}", flush=True)
    print(f"Using device: {get_device()}", flush=True)

    # Running experiment with arguments from terminal and configuration file
    experiment_module_name, experiment_class_name = split_only_last(arguments.experiment)
    experiment_module = importlib.import_module(experiment_module_name)
    Experiment = getattr(experiment_module, experiment_class_name)
    experiment = Experiment()
    experiment.run(arguments)