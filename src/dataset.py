import torch
import random
from enum import IntEnum
from numpy import np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from abc import abstractmethod

class Base_Dataset(Dataset): 
    def __init__(self): 
        pass
    
    @abstractmethod
    def create_image(self, index, quad_index = 0): 
        raise NotImplementedError("Create image function is not implemented!")

    @abstractmethod
    def create_image_augmented(self, index, quad_index = 0):
        raise NotImplementedError("Create image augmented function is not implemented!")

    @abstractmethod
    def create_image_beta(self, index): 
        raise NotImplementedError("Create image beta function is not implemented!")

    def create_image_gamma(self, index): 
        raise NotImplementedError("Create image gamma function is not implemented!")

    @abstractmethod
    def get_label_names(self): 
        raise NotImplementedError("get label names function is not implemented!")

    def save_sample(self, name):
        label, image = self.__getitem__(np.random.randint(0, len(self)))
        image = image.cpu().numpy()

        path_first_part = f"data/results_data/{name}"
        path = f"{path_first_part}/single_sample.png"


        image = image.reshape(image.shape[1], image.shape[1], 3 if image.shape[0] == 3 else 1) * 255
    
        if image.shape[2] == 1: 
            image = image.reshape(image.shape[0], image.shape[1])

        image = image.astype(np.uint8)

        Image.fromarray(image).save(path)

class Quad_Dataset_Sampling_Mode(IntEnum):
    RANDOM = 0
    PERMUTATIONS = 1

class QuadDataset(Dataset): 
    def __init__(self, base_dataset : Base_Dataset, sampling_mode : Quad_Dataset_Sampling_Mode):
        self.base_dataset = base_dataset
        self.sampling_mode = sampling_mode

        self.length = len(self.base_dataset) if self.sampling_mode == Quad_Dataset_Sampling_Mode.RANDOM else len(self.base_dataset) * len(self.base_dataset)

        self.indices = [] 
        self.other_indices = []

        for i in range(0, self.length): 
            if self.sampling_mode == Quad_Dataset_Sampling_Mode.RANDOM:
                self.indices.append(i)
                self.other_indices.append(random.randint(0, self.length - 1))
            else: 
                self.indices.append(i % len(self.base_dataset))
                self.other_indices.append(i // len(self.base_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        image_1 = self.base_dataset.create_image(self.indices[index], 0)
        image_2 = self.base_dataset.create_image(self.other_indices[index], 1)
        image_3 = self.base_dataset.create_image_augmented(self.indices[index], 2)
        image_4 = self.base_dataset.create_image_augmented(self.other_indices[index], 3)

        return torch.stack((image_1, image_2, image_3, image_4), dim=3)

    def save_sample(self, name): 
        images = self[np.random.randint(0, len(self) - 1)].cpu().numpy()

        path_first_part = f"data/results_data/{name}"
        path_1 = f"{path_first_part}/q_1.png"
        path_2 = f"{path_first_part}/q_1_augmented.png"
        path_3 = f"{path_first_part}/q_2.png"
        path_4 = f"{path_first_part}/q_2_augmented.png"

        image_1 = images[:, :, :, 0].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_2 = images[:, :, :, 2].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_3 = images[:, :, :, 1].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_4 = images[:, :, :, 3].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255

        
        if image_1.shape[2] == 1: 
            image_1 = image_1.reshape(image_1.shape[0], image_1.shape[1])
            image_2 = image_2.reshape(image_1.shape[0], image_1.shape[1])
            image_3 = image_3.reshape(image_1.shape[0], image_1.shape[1])
            image_4 = image_4.reshape(image_1.shape[0], image_1.shape[1])

        image_1 = image_1.astype(np.uint8)
        image_2 = image_2.astype(np.uint8)
        image_3 = image_3.astype(np.uint8)
        image_4 = image_4.astype(np.uint8)

        Image.fromarray(image_1).save(path_1)
        Image.fromarray(image_2).save(path_2)
        Image.fromarray(image_3).save(path_3)
        Image.fromarray(image_4).save(path_4)

class DirectionalQuadDataset(QuadDataset):
    def __init__(self, base_dataset : Base_Dataset, sampling_mode : Quad_Dataset_Sampling_Mode):
        super(DirectionalQuadDataset, self).__init__(base_dataset, sampling_mode)
    
    def __getitem__(self, index): 
        quad = super().__getitem__(index) 

        gamma_image = self.base_dataset.create_image_gamma(index)
        beta_image = self.base_dataset.create_image_beta(index)

        gamma_image = gamma_image.reshape(gamma_image.shape + (1, ))
        beta_image = beta_image.reshape(beta_image.shape + (1, ))

        return torch.cat((quad, beta_image, gamma_image), dim=3)

    def save_sample(self, name): 
        images = self[np.random.randint(0, len(self))].cpu().numpy()

        path_first_part = f"data/results_data/{name}"
        path_1 = f"{path_first_part}/q_1.png"
        path_2 = f"{path_first_part}/q_1_augmented.png"
        path_3 = f"{path_first_part}/q_2.png"
        path_4 = f"{path_first_part}/q_2_augmented.png"
        path_5 = f"{path_first_part}/beta.png"
        path_6 = f"{path_first_part}/gamma.png"

        image_1 = images[:, :, :, 0].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_2 = images[:, :, :, 2].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_3 = images[:, :, :, 1].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_4 = images[:, :, :, 3].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_5 = images[:, :, :, 4].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        image_6 = images[:, :, :, 5].reshape(images.shape[1], images.shape[1], 3 if images.shape[0] == 3 else 1) * 255
        
        if image_1.shape[2] == 1: 
            image_1 = image_1.reshape(image_1.shape[0], image_1.shape[1])
            image_2 = image_2.reshape(image_1.shape[0], image_1.shape[1])
            image_3 = image_3.reshape(image_1.shape[0], image_1.shape[1])
            image_4 = image_4.reshape(image_1.shape[0], image_1.shape[1])
            image_5 = image_5.reshape(image_1.shape[0], image_1.shape[1])
            image_6 = image_6.reshape(image_1.shape[0], image_1.shape[1])

        image_1 = image_1.astype(np.uint8)
        image_2 = image_2.astype(np.uint8)
        image_3 = image_3.astype(np.uint8)
        image_4 = image_4.astype(np.uint8)
        image_5 = image_5.astype(np.uint8)
        image_6 = image_6.astype(np.uint8)

        Image.fromarray(image_1).save(path_1)
        Image.fromarray(image_2).save(path_2)
        Image.fromarray(image_3).save(path_3)
        Image.fromarray(image_4).save(path_4)
        Image.fromarray(image_5).save(path_5)
        Image.fromarray(image_6).save(path_6)