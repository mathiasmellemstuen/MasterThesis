import torch_optimizer as optim
import torch
import time
import numpy as np
import importlib
import matplotlib.pyplot as plt
from utils.print_epoch_info import print_epoch_info
from utils.get_device import get_device
from utils.label import * 
from utils.label_type import *
from torchsummary import summary
from networks import save_model, load_model
from save import save_numpy_array_to_csv
from tqdm import tqdm
from utils.split_only_last import split_only_last
from pathlib import Path
from csv_saver import *

def train_model(model, optimizer, dataloader, epochs, name, save_interval):
    
    start_time = time.time()
    
    for epoch in range(epochs + 1): 

        last_epoch_start_time = time.time()

        accumulated_batch_loading_time = 0
        
        batch_load_start_time = time.time()

        with tqdm(total=len(dataloader)) as progress_bar:
            for i, image_batch in enumerate(dataloader):
                
                # Moving to the GPU here instead of in the dataloader
                image_batch = image_batch.to(get_device())

                batch_load_end_time = time.time()

                loss = model(image_batch)
                csv_input("LOSS", loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)

        loss = np.mean(list(csv_retrieve_last("LOSS")))
        accumulated_batch_loading_time += batch_load_end_time - batch_load_start_time
        print_epoch_info(epoch=epoch, epochs=epochs, start_time=start_time, last_epoch_start_time=last_epoch_start_time, accumulated_batch_loading_time = accumulated_batch_loading_time, losses=loss, lr=optimizer.param_groups[-1]["lr"])

        csv_input("EPOCH", epoch)
        csv_step()

        if epoch % save_interval == 0 or epoch == epochs:

            save_model(model,f"data/saved_models/{name}/{name}_epoch_{epoch}.pt")
            csv_save(name, "train_data")

        batch_load_start_time = time.time()