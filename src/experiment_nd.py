from utils.get_device import get_device
from quad_network import QuadNetwork
from torch.utils.data import DataLoader
from experiment import Experiment
from networks import load_model
from dataset import QuadDataset
from utils.split_only_last import split_only_last
import torch_optimizer as optim
import torch
import importlib

class Experiment_ND(Experiment): 
    def __init__(self):
        pass

    def run(self, arguments):
        
        print("Running experiment")

        # Setting up and creating new folders for this spesific experiment
        super().__init__(arguments.experiment_name, arguments.experiment_display_name)
        
        networks = []

        for network_name in arguments.networks: 
            network_module_name, network_class_name = split_only_last(network_name)
            network_module = importlib.import_module(network_module_name)
            Network = getattr(network_module, network_class_name)
            networks.append(Network(*arguments.network_arguments))

        encoder_module_name, encoder_class_name = split_only_last(arguments.encoder)
        encoder_module = importlib.import_module(encoder_module_name)
        Encoder = getattr(encoder_module, encoder_class_name)
        encoder = Encoder(*arguments.encoder_arguments)

        dataset_module_name, dataset_class_name = split_only_last(arguments.dataset)
        dataset_module = importlib.import_module(dataset_module_name)
        Dataset = getattr(dataset_module, dataset_class_name)

        wrapper_module_name, wrapper_module_class_name = split_only_last(arguments.wrapper_module)
        wrapper_module = importlib.import_module(wrapper_module_name)
        WrapperModule = getattr(wrapper_module, wrapper_module_class_name)
        network = WrapperModule(encoder, networks, *arguments.wrapper_module_arguments)

        wrapper_dataset_module_name, wrapper_dataset_module_class_name = split_only_last(arguments.wrapper_dataset_module)
        wrapper_dataset_module = importlib.import_module(wrapper_dataset_module_name)
        WrapperDataset = getattr(wrapper_dataset_module, wrapper_dataset_module_class_name)

        dataset = WrapperDataset(Dataset(*arguments.dataset_arguments), *arguments.wrapper_dataset_module_arguments)

        network = network.to(get_device())

        dataset.save_sample(arguments.experiment_name)

        print(f"Using device: {get_device()}, dataset: {arguments.dataset} with {len(dataset)} samples", flush=True)

        # If training is enabled, create the dataloader, optimizer and run the training function spesified from arguments
        if arguments.train: 
            print("Training model...")
            dataloader = DataLoader(dataset, batch_size=arguments.batch_size, num_workers=arguments.dataloader_workers, shuffle=True, drop_last=True)

            trainer_module_name, trainer_function_name = split_only_last(arguments.trainer_name)
            trainer_module = importlib.import_module(trainer_module_name)
            trainer = getattr(trainer_module, trainer_function_name)

            optimizer = optim.AdaBelief(network.parameters(), lr=arguments.learning_rate)

            trainer(network, optimizer, dataloader, arguments.epochs, arguments.experiment_name, arguments.save_interval, *arguments.trainer_arguments)

        # Load the already trained model 
        model_path = f"data/saved_models/{arguments.experiment_name}/{arguments.experiment_name}_epoch_{arguments.epochs}.pt"
        print(f"Loading pretrained model from {model_path}")
        load_model(network, model_path)

        print("Testing model...")

        network.eval() 

        for element, args in arguments.test_functions:
            test_function_module_name, test_function_name = split_only_last(element)
            test_function_module = importlib.import_module(test_function_module_name)
            test_func = getattr(test_function_module, test_function_name)

            test_func(arguments.experiment_name, network, dataset, *args)