from geometric_shape_generator.geometric_shape_generator import generate_geometric_shape
from geometric_shape_generator.geometric_shapes import GeometricShape
from geometric_shape_generator.color_mode import  ColorMode
from enum import IntEnum
from typing import List
from torch.utils.data import Dataset
from abc import abstractmethod
from utils.noise import noise_image
from utils.color_gradient import color_gradient_min_red_max_blue, color_gradient_min_red_max_blue_inverse
from PIL import Image
from skimage.morphology import skeletonize
from tqdm import tqdm
from pathlib import Path
from os.path import exists
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2 as transforms
import torch
import torchvision.transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import albumentations
import os
import cv2
import torchvision.datasets as datasets
import datasets as ds

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

class Greyscale_Rectangles_Dataset(Base_Dataset):
    def __init__(self, min_size, max_size, image_size, noise_sigma):
        self.min_size = min_size
        self.max_size = max_size
        self.image_size = image_size
        self.noise_sigma = noise_sigma
        
        self.length = (self.max_size - self.min_size)
        self.sizes = list(range(self.min_size, self.max_size))

    def create_image_beta(self, index):
        return self.__create_image(min(self.sizes))
    
    def create_image_gamma(self, index):
        return self.__create_image(max(self.sizes))

    def create_image(self, index, quad_index = 0):
        return self.__create_image(self.sizes[index])

    def create_image_augmented(self, index, quad_index = 0):
        return self.__create_image(self.sizes[index])

    def __create_image(self, size):
        current_position = (random.randint(size, self.image_size[0] - size), random.randint(size, self.image_size[1] - size))
        image = generate_geometric_shape(center_position=current_position, image_size=self.image_size, color=255, color_mode=ColorMode.GRAYSCALE, size=size, shape_type=GeometricShape.RECTANGLE)
        image = image.reshape((image.shape[2] if len(image.shape) > 2 else 1, image.shape[1], image.shape[0]))

        image = image.astype(np.float32)
        image /= 255.0
        image = torch.tensor(image)
        image = noise_image(image, self.noise_sigma)

        return image

    def __len__(self):
        return self.length

    def __getitem__(self, index): 
        return [self.sizes[index]], self.create_image(index)
    
    def get_label_names(self):
        return ["SIZE"]

class Both_RGB_Size_Rectangles_Dataset(Greyscale_Rectangles_Dataset):
    def __init__(self, min_size, max_size, image_size, noise_sigma):
        super().__init__(min_size, max_size, image_size, noise_sigma)

        self.colors = [color_gradient_min_red_max_blue(0, 1.0, x / self.length) for x in range(0, self.length)]

        self.last_size = 0

        self.color_index_1 = 0
        self.color_index_2 = 0

    def create_image_beta(self, index):
        return self.private_create_image(min(self.sizes), (0, 0, 0))

    def create_image_gamma(self, index):
        return self.private_create_image(max(self.sizes), (0, 0, 255))

    def create_image(self, index, quad_index = 0):
        
        
        if quad_index == 0: 
            self.color_index_1 = np.random.randint(0, len(self.colors) - 1)
            c_ind = self.color_index_1
        
        if quad_index == 1:
            self.color_index_2 = np.random.randint(0, len(self.colors) - 1)
            c_ind = self.color_index_2

        return self.private_create_image(self.sizes[index], self.colors[c_ind])
    
    def create_image_augmented(self, index, quad_index = 0):
        # color_index = (index + self.color_index_displacement) % (len(self.colors) - 1)

        if quad_index == 2: 
            return self.private_create_image(self.sizes[index], self.colors[self.color_index_1])
        
        if quad_index == 3: 
            return self.private_create_image(self.sizes[index], self.colors[self.color_index_2])

    def create_random_RGB_color(self): 
        self.last_random_color = (random.randint(1, 255), random.randint(0, 255), random.randint(0, 255))
        return self.last_random_color

    def private_create_image(self, size, color):
        current_position = (random.randint(size, self.image_size[0] - size), random.randint(size, self.image_size[1] - size))
        image = generate_geometric_shape(center_position=current_position, image_size=self.image_size, color=color, color_mode=ColorMode.RGB, size=size, shape_type=GeometricShape.RECTANGLE)
        image = image.reshape((image.shape[2] if len(image.shape) > 2 else 1, image.shape[1], image.shape[0]))

        image = image.astype(np.float32)
        image /= 255.0
        image = torch.tensor(image)
        image = noise_image(image, self.noise_sigma)

        return image

    def __getitem__(self, index): 
        # image = self.create_image(index)
        col = self.colors[np.random.randint(0, len(self.colors) - 1)]
        image = self.private_create_image(self.sizes[index], col)
        
        return [self.sizes[index], col[0], col[1], col[2]], image

    def get_label_names(self):
        return ["SIZE", "COLOR_R", "COLOR_G", "COLOR_B"]

class Caltech_Grid_Dastaset(Base_Dataset):
    def __init__(self, caltech_path, train, grid_size, image_dimension, min_num_shuffles, max_num_shuffles, noise_sigma):

        self.caltech_path = caltech_path
        self.image_dimension = image_dimension
        self.grid_size = grid_size
        self.grid_amount = self.grid_size[0] * self.grid_size[1]

        self.piece_x_size = int(np.floor(self.image_dimension / self.grid_size[0]))
        self.piece_y_size = int(np.floor(self.image_dimension / self.grid_size[1]))

        self.min_num_shuffles = min_num_shuffles
        self.max_num_shuffles = max_num_shuffles

        self.noise_sigma = noise_sigma

        if max_num_shuffles > np.ceil(self.grid_amount / 2):
            raise Exception(f"The max_num_shuffles({max_num_shuffles}) needs to be smaller or equal to {np.ceil(self.grid_amount / 2)}")

        self.caltech = datasets.Caltech256(root=self.caltech_path, download=True)
        self.image_dimension = image_dimension

        self.first_shuffle_amount = 0
        self.second_shuffle_amount = 0

        self.augmentations = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
        ])

    def create_correct_configuration_matrix(self): 
        mat = np.zeros(self.grid_size)

        index = 0
        for y in range(self.grid_size[0]): 
            for x in range(self.grid_size[1]): 
                mat[y, x] = index
                index += 1

        return mat

    def create_shuffle_configuration(self, num_shuffles):

        configuration_matrix = self.create_correct_configuration_matrix()
        all_indices = []

        for y in range(self.grid_size[0]): 
            for x in range(self.grid_size[1]): 
                all_indices.append((y, x))

        for _ in range(num_shuffles): 
            random_ind = np.random.randint(0, len(all_indices) - 1)
            random_from = all_indices[random_ind]
            del all_indices[random_ind]

            if len(all_indices) != 0: 
                random_ind = np.random.randint(0, len(all_indices) - 1)
                random_to = all_indices[random_ind]
                del all_indices[random_ind]
            else:
                return configuration_matrix

            temp_value = configuration_matrix[random_to[0], random_to[1]]
            configuration_matrix[random_to[0], random_to[1]] = configuration_matrix[random_from[0], random_from[1]]
            configuration_matrix[random_from[0], random_from[1]] = temp_value

        return configuration_matrix

    def shuffle_image(self, image, shuffle_configuration):
        image_copy = image.copy()

        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]): 
                to_x = shuffle_configuration[y, x] % self.grid_size[0]
                to_y = shuffle_configuration[y, x] // self.grid_size[1]

                from_x = x
                from_y = y

                if to_x == from_x and to_y == from_y:
                    continue
                
                from_y_min = int(from_y * self.piece_y_size)
                from_y_max = int((from_y + 1) * self.piece_y_size)

                from_x_min = int(from_x * self.piece_x_size)
                from_x_max = int((from_x + 1) * self.piece_x_size)

                to_y_min = int(to_y * self.piece_y_size)
                to_y_max = int((to_y + 1) * self.piece_y_size)

                to_x_min = int(to_x * self.piece_x_size)
                to_x_max = int((to_x + 1) * self.piece_x_size)

                image[to_y_min : to_y_max, to_x_min : to_x_max, :] = image_copy[from_y_min : from_y_max, from_x_min : from_x_max, :]
                image[from_y_min : from_y_max, from_x_min : from_x_max, :] = image_copy[to_y_min : to_y_max, to_x_min : to_x_max, :]


        return image

    def to_tensor(self, image):
        image = image.reshape((image.shape[2] if len(image.shape) > 2 else 1, image.shape[1], image.shape[0]))

        image = image.astype(np.float32)
        image /= 255.0
        image = torch.tensor(image)
        image = noise_image(image, self.noise_sigma)
        return image

    def _create_image(self, index, shuffle_configuration):
        image = cv2.resize(np.array(self.caltech[index][0]), dsize=(self.image_dimension, self.image_dimension))
        
        if len(image.shape) != 3 or image.shape[2] != 3: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = self.shuffle_image(image, shuffle_configuration)

        return image

    def create_image_beta(self, index):
        shuffle_config = self.create_shuffle_configuration(self.min_num_shuffles)
        return self.to_tensor(self._create_image(np.random.randint(0, len(self)), shuffle_config))

    def create_image_gamma(self, index):
        shuffle_config = self.create_shuffle_configuration(self.max_num_shuffles)
        return self.to_tensor(self._create_image(np.random.randint(0, len(self)), shuffle_config))

    def create_image(self, index, quad_index = 0, tensor = True):

        if quad_index == 0:
            shuffles = list(range(self.min_num_shuffles, self.max_num_shuffles))
            self.first_shuffle_amount = np.random.choice(shuffles)
            shuffles.remove(self.first_shuffle_amount)
            self.second_shuffle_amount = np.random.choice(shuffles)

        shuffle_amount = self.first_shuffle_amount if (quad_index == 0 or quad_index == 2) else self.second_shuffle_amount

        image = self._create_image(index % len(self.caltech), self.create_shuffle_configuration(shuffle_amount))
        return self.to_tensor(image) if tensor else image

    def create_image_augmented(self, index, quad_index = 0):
        image = self.create_image(index, quad_index, False)
        image = self.augmentations(image)
        image = np.array(image)

        return self.to_tensor(image)

    def __len__(self):
        return len(self.caltech)

    def __getitem__(self, index):
        image = self.create_image(index)
        return [self.caltech[index][1], self.first_shuffle_amount], image

    def get_label_names(self):
        return ["SHUFFLE_DISTANCE", "CALTECH_CLASS"]
    


class Caltech_Grid_Hamming_Distance_Dataset(Caltech_Grid_Dastaset):
    def __init__(self, caltech_path, train, grid_size, image_dimension, min_num_shuffles, max_num_shuffles, num_configurations, noise_sigma):
        super().__init__(caltech_path, train, grid_size, image_dimension, min_num_shuffles, max_num_shuffles, noise_sigma)
        
        self.num_configurations = num_configurations

        self.all_distances = []
        self.all_configurations = []

        self.pre_create_configurations()
        
        self.unique_distances = np.unique(self.all_distances)

        self.all_distances = np.array(self.all_distances, dtype=np.uint8)

        self.first_distance = 0
        self.second_distance = 0

    def calculate_distance(self, configuration):
        # Hamming distance
        
        distance = 0
        #correct_matrix = self.create_correct_configuration_matrix()

        for y in range(self.grid_size[0]): 
            for x in range(self.grid_size[1]):

                correct_x = configuration[y, x] % self.grid_size[0]
                correct_y = configuration[y, x] // self.grid_size[1]
                distance += np.abs(x - correct_x)
                distance += np.abs(y - correct_y)

        distance = distance / 2

        return distance

    def pre_create_configurations(self):
        configurations_per_shuffles = self.num_configurations // (self.max_num_shuffles - self.min_num_shuffles)

        with tqdm(total=(self.max_num_shuffles - self.min_num_shuffles) * configurations_per_shuffles) as progress_bar:
            progress_bar.set_description("Dataset creation: Creating unique configurations")
            for i in range(self.min_num_shuffles, self.max_num_shuffles):
                for _ in range(configurations_per_shuffles):

                    configuration = self.create_shuffle_configuration(i)

                    # Trying without this just to create a uniform distribution of the shuffles
                    # We don't want equal samples in the list
                    #if configuration in self.all_configurations:
                    #        continue
                    
                    distance = self.calculate_distance(configuration)

                    self.all_distances.append(distance)
                    self.all_configurations.append(configuration)
                    progress_bar.update(1)
    
    def get_configuration_with_distance(self, distance):

        indices_with_distance = np.where(self.all_distances ==  distance)
        random_index = indices_with_distance[0][np.random.randint(0, len(indices_with_distance)) - 1]
        
        return self.all_configurations[random_index]


    def create_image_beta(self, index):

        shuffle_configuration = self.get_configuration_with_distance(np.min(self.unique_distances))
        return self.to_tensor(self._create_image(np.random.randint(0, len(self.caltech)), shuffle_configuration))

    def create_image_gamma(self, index):

        shuffle_configuration = self.get_configuration_with_distance(np.max(self.unique_distances))
        return self.to_tensor(self._create_image(np.random.randint(0, len(self.caltech)), shuffle_configuration))

    def create_image(self, index, quad_index = 0, tensor = True):

        if quad_index == 0:
            distances = np.copy(self.unique_distances)
            self.first_distance = self.all_distances[index]
            distances = list(distances)
            distances.remove(self.first_distance)
            self.second_distance = np.random.choice(distances)

        distance = self.first_distance if (quad_index == 0 or quad_index == 2) else self.second_distance

        configuration = self.all_configurations[index]

        if quad_index == 2:
            configuration = self.get_configuration_with_distance(self.first_distance)

        elif quad_index == 1 or quad_index == 3:
            configuration = self.get_configuration_with_distance(distance)

        image = self._create_image(index % (len(self.caltech) - 1), configuration)

        return self.to_tensor(image) if tensor else image

    def __len__(self):
        return len(self.all_configurations)
    
    def __getitem__(self, index):
        caltech_index = np.random.randint(0, len(self.caltech))
        image = self._create_image(caltech_index, self.all_configurations[index])
        return [self.all_distances[index], np.sum(image), self.caltech[caltech_index][1]], self.to_tensor(image)

    def get_label_names(self):
        return ["SHUFFLE_DISTANCE", "PIXEL_SUM", "CALTECH_CLASS"]

class Caltech_Grid_Neighbour_Distance_Dataset(Caltech_Grid_Hamming_Distance_Dataset): 
    def __init__(self, caltech_path, train, grid_size, image_dimension, min_num_shuffles, max_num_shuffles, num_configurations, noise_sigma):
        
        Caltech_Grid_Dastaset.__init__(self, caltech_path, train, grid_size, image_dimension, min_num_shuffles, max_num_shuffles, noise_sigma)
        
        self.num_configurations = num_configurations

        self.all_distances = []
        self.all_configurations = []

        self.correct_matrix = self.create_correct_configuration_matrix()
        self.pre_create_configurations()
        
        self.unique_distances = np.unique(self.all_distances)

        self.all_distances = np.array(self.all_distances, dtype=np.uint8)

        self.first_distance = 0
        self.second_distance = 0

    def calculate_distance(self, configuration):
        """
        Assuming that we use 3x3 grid, the self.correct_matrix is a ndarray with the following: 
        
        0 1 2 
        3 4 5
        6 7 8

        This represents a image where 0 patches have been swapped.

        the configuration argument is another 3x3 grid, that shows us what cells in the 3x3 grid that have been swapped. Example
        patch 0 -> 8 and 3 -> 4 have been swapped the configuration would be:
        8 1 2
        4 3 5
        6 7 0
        """
        score = 0

        for i in range(self.correct_matrix.shape[0]):
            for j in range(self.correct_matrix.shape[1]):

                current_id = configuration[i, j]
                o_i, o_j = np.where(self.correct_matrix == current_id)

                o_up_neighbour = self.correct_matrix[o_i - 1, o_j] if o_i >= 1 else None
                o_down_neighbour = self.correct_matrix[o_i + 1, o_j] if o_i < self.correct_matrix.shape[0] - 1 else None
                o_left_neighbour = self.correct_matrix[o_i, o_j - 1] if o_j >= 1 else None
                o_right_neighbour = self.correct_matrix[o_i, o_j + 1] if o_j < self.correct_matrix.shape[0] - 1 else None
                
                c_up_neighbour = configuration[i - 1, j] if i >= 1 else None
                c_down_neighbour = configuration[i + 1, j] if i < configuration.shape[0] - 1 else None
                c_left_neighbour = configuration[i, j - 1] if j >= 1 else None
                c_right_neighbour = configuration[i, j + 1] if j < configuration.shape[0] - 1 else None

                score += 1 if o_up_neighbour != c_up_neighbour else 0
                score += 1 if o_down_neighbour != c_down_neighbour else 0
                score += 1 if o_left_neighbour != c_left_neighbour else 0
                score += 1 if o_right_neighbour != c_right_neighbour else 0

        return score

class Color_Gradient_Rectangles_Dataset(Greyscale_Rectangles_Dataset):
    def __init__(self, min_size, max_size, image_size, noise_sigma):
        super().__init__(min_size, max_size, image_size, noise_sigma)
        self.colors = [color_gradient_min_red_max_blue(0, 1.0, x / self.length) for x in range(0, self.length)]
        self.last_size = 0

    def create_image_beta(self, index):
        return self.private_create_image((255, 0, 0))

    def create_image_gamma(self, index):
        return self.private_create_image((0, 0, 255))

    def create_image(self, index, quad_index = 0):
        return self.private_create_image(self.colors[index])

    def create_image_augmented(self, index, quad_index = 0):
        return self.private_create_image(self.colors[index])

    def private_create_image(self, color):
        size = random.randint(self.min_size,  self.max_size)
        self.last_size = size
        current_position = (random.randint(size, self.image_size[0] - size), random.randint(size, self.image_size[1] - size))
        image = generate_geometric_shape(center_position=current_position, image_size=self.image_size, color=color, color_mode=ColorMode.RGB, size=size, shape_type=GeometricShape.RECTANGLE)
        image = image.reshape((image.shape[2] if len(image.shape) > 2 else 1, image.shape[1], image.shape[0]))

        image = image.astype(np.float32)
        image /= 255.0
        image = torch.tensor(image)
        image = noise_image(image, self.noise_sigma)

        return image
    
    def __getitem__(self, index): 
        image = self.create_image(index)
        gradient_value = (index / self.length)
        return [self.last_size, gradient_value, self.colors[index][0], self.colors[index][1], self.colors[index][2]], image
    
    def get_label_names(self):
        return ["SIZE", "GRADIENT_VALUE", "COLOR_R", "COLOR_G", "COLOR_B"]

class Gap_Mode(IntEnum): 
    MIDDLE = 0
    LEFT = 1
    RIGHT = 2
    UNIFORM_RANDOM = 3
    INTERVAL = 4

class Greyscale_Rectangles_Gaps_Dataset(Greyscale_Rectangles_Dataset): 
    def __init__(self, min_size, max_size, image_size, noise_sigma, gap_mode : Gap_Mode, gap_size, amount_of_gaps):
        super().__init__(min_size, max_size, image_size, noise_sigma)
        self.gap_mode = gap_mode
        self.gap_size = gap_size
        self.amount_of_gaps = amount_of_gaps

        self.included = []

        if self.gap_mode == Gap_Mode.MIDDLE: 
            self.included = list(range(self.min_size, (self.max_size // 2) - (self.gap_size // 2))) + list(range((self.max_size // 2) + (self.gap_size // 2), self.max_size))
        
        if self.gap_mode == Gap_Mode.LEFT: 
            self.included = list(range(self.min_size + self.gap_size, self.max_size))
        
        if self.gap_mode == Gap_Mode.RIGHT: 
            self.included = list(range(self.min_size, self.max_size - self.gap_size))

        if self.gap_mode == Gap_Mode.UNIFORM_RANDOM:
            self.included = list(range(self.min_size, self.max_size))
            
            for _ in range(amount_of_gaps): 
                random_index = np.random.randint(0, len(self.included) - self.gap_size)
                indices = [random_index + j for j in range(self.gap_size)]

                for index in indices: 
                    self.included.pop(index)
        
        if self.gap_mode == Gap_Mode.INTERVAL:
            self.included = list(range(self.min_size, self.max_size))
            interval_step = (self.max_size - self.min_size) // self.amount_of_gaps

            current = 0
            while current < len(self.included) - self.gap_size: 
                for _ in range(self.gap_size): 
                    self.included.pop(current)

                current += interval_step
        
        self.excluded = list(set(self.sizes) - set(self.included))
        
        self.excluded_length = len(self.excluded)
        self.included_length = len(self.included)
        self.set_mode(True)
    
    def set_mode(self, included):
        if included: 
            self.sizes = self.included
        else: 
            self.sizes = self.excluded

    def __len__(self): 
        return len(self.sizes)

class Masked_3D_Shapes_Dataset(Base_Dataset):
    def __init__(self, dataset_path, noise_sigma):
        self.dataset_path = dataset_path
        self.noise_sigma = noise_sigma

        self.all_folders = os.listdir(f"{self.dataset_path}/colors/")

        if ".DS_Store" in self.all_folders: 
            self.all_folders.remove(".DS_Store")

        self.angles = []

        with open(f"{self.dataset_path}/angles.txt") as f: 
            lines = f.readlines()
            self.angles = [float(x.replace(",", ".")) for x in lines]

        self.wall_masks = np.zeros((256, 256, 3, len(self.angles)), dtype=np.uint8)
        self.floor_masks = np.zeros((256, 256, 3, len(self.angles)), dtype=np.uint8)
        self.cube_masks = np.zeros((256, 256, 3, len(self.angles)), dtype=np.uint8)

        for i in range(len(self.angles)): 
            self.wall_masks[:, :, :, i] = self.__load_file_to_np(f"{self.dataset_path}/masks/wall/{i}.png")
            self.floor_masks[:, :, :, i] = self.__load_file_to_np(f"{self.dataset_path}/masks/floor/{i}.png")
            self.cube_masks[:, :, :, i] = self.__load_file_to_np(f"{self.dataset_path}/masks/cube/{i}.png")

        self.all_images = np.zeros((256, 256, 3, len(self.angles), len(self.all_folders)), dtype=np.uint8)

        for i, folder_name in enumerate(self.all_folders): 
            for j in range(len(self.angles)):
                full_file_path = f"{self.dataset_path}/colors/{folder_name}/{j}.png"
                self.all_images[:, :, :, j, i] = self.__load_file_to_np(full_file_path)

        self.length = len(self.angles) * len(self.all_folders)

    def __len__(self): 
        return self.length
    
    def __getitem__(self, index): 
        return [self.angles[index % len(self.angles)]], self.create_image(index)
    
    def get_label_names(self):
        return ["ANGLE"]

    def create_image(self, index, quad_index = 0):
        frame_index = index % len(self.angles)
        folder_index = index // len(self.angles)
        return self._create_image(frame_index, folder_index)

    def create_image_augmented(self, index, quad_index = 0):
        frame_index = index % len(self.angles)
        folder_index = random.randint(0, len(self.all_folders) - 1)

        return self._create_image(frame_index, folder_index)

    def __augment_hue_on_image_with_masks(self, cube_mask, floor_mask, wall_mask, image):

        wall = cv2.bitwise_and(image, wall_mask)
        cube = cv2.bitwise_and(image, cube_mask)
        floor = cv2.bitwise_and(image, floor_mask) - cube
        sky = cv2.bitwise_and(image, cv2.bitwise_not(wall_mask + floor_mask + cube_mask))

        wall = self.__shift_hue(wall, random.randint(0, 179))
        cube = self.__shift_hue(cube, random.randint(0, 179))
        floor = self.__shift_hue(floor, random.randint(0, 179))
        
        image = wall + floor + cube + sky

        return image

    def __shift_hue(self, image, hue_value):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        hsv_image = hsv_image.copy()
        h, s, v = cv2.split(hsv_image)

        diff_color = h - hue_value
        h_new = np.mod(h + diff_color, 179).astype(np.uint8)

        hsv_combined = cv2.merge([h_new, s, v])

        rgb_image = cv2.cvtColor(hsv_combined, cv2.COLOR_HSV2RGB_FULL)

        return rgb_image

    def _create_image(self, frame_index, folder_index): 

        cube_mask = self.cube_masks[:, :, :, frame_index]
        floor_mask = self.floor_masks[:, :, :, frame_index]
        wall_mask = self.wall_masks[:, :, :, frame_index]

        image = self.all_images[:, :, :, frame_index, folder_index]

        image = self.__augment_hue_on_image_with_masks(cube_mask, floor_mask, wall_mask, image)

        image = image.astype(np.float32)
        image /= 255.0
        image = image.reshape(3, 256, 256)
        image = torch.tensor(image)
        image = noise_image(image, self.noise_sigma)

        return image

    def __load_file_to_np(self, file_path):

        pil_image = Image.open(file_path)
        image = np.array(pil_image, dtype=np.uint8).reshape(256, 256, 3)
        pil_image.close()

        return image

class Masked_3D_Shapes_Gap_Dataset(Masked_3D_Shapes_Dataset): 
    def __init__(self, dataset_path, noise_sigma, gap_mode : Gap_Mode, gap_size, amount_of_gaps): 
        super().__init__(dataset_path, noise_sigma)

        self.gap_mode = gap_mode
        self.gap_size = gap_size
        self.amount_of_gaps = amount_of_gaps

        self.included = []
        self.excluded = []
        
        n = len(self.angles)
        n_half = n // 2

        if self.gap_mode == Gap_Mode.MIDDLE:
            self.included = list(range(0, n_half - (self.gap_size // 2))) + list(range(n_half + (self.gap_size // 2), n))
            self.excluded = list(range(n_half - (self.gap_size // 2), n_half + self.gap_size // 2))
        
        if self.gap_mode == Gap_Mode.LEFT:
            self.excluded = list(range(0, self.gap_size))
            self.included = list(range(self.gap_size, n))

        if self.gap_mode == Gap_Mode.RIGHT:
            self.excluded = list(range(n - self.gap_size, n))
            self.included = list(range(0, n - self.gap_size))

        if self.gap_mode == Gap_Mode.UNIFORM_RANDOM:
            self.included = list(range(0, n))
            
            for i in range(amount_of_gaps): 
                random_index = np.random.randint(0, n - gap_size)
                indices = [random_index + j for j in range(self.gap_size)]

                for index in indices: 
                    self.excluded.append(self.included[index])
                    self.included.pop(index)

        if self.gap_mode == Gap_Mode.INTERVAL: 
            self.included = list(range(0, n))
            interval_step = n // self.amount_of_gaps
            
            current = 0
            while current < len(self.included) - self.gap_size: 
                for i in range(self.gap_size): 
                    self.excluded.append(self.included[current])
                    self.included.pop(current)
                    
                current += interval_step - 1
        
        self.excluded_length = len(self.excluded) * len(self.all_folders)
        self.included_length = len(self.included) * len(self.all_folders)

        self.current_angles = []
        self.set_mode(True)

    def set_mode(self, included): 
        if included: 
            self.current_angles = self.included
            self.length = len(self.included) * len(self.all_folders)
        else: 
            self.current_angles = self.excluded
            self.length = len(self.excluded) * len(self.all_folders)

    def __getitem__(self, index): 
            return [self.angles[self.current_angles[index % len(self.current_angles)]]], self.create_image(index)

    def create_image(self, index, quad_index = 0):
        frame_index = self.current_angles[index % len(self.current_angles)]
        folder_index = index // len(self.current_angles)

        return self._create_image(frame_index, folder_index)

    def create_image_augmented(self, index, quad_index = 0):
        frame_index = self.current_angles[index % len(self.current_angles)]
        folder_index = random.randint(0, len(self.all_folders) - 1)

        return self._create_image(frame_index, folder_index)

class MNIST_Dataset(Base_Dataset):
    def __init__(self, mnist_path, dtd_path, train, use_textures, grayscale_textures, texture_class_whitelist, image_dimension = 256, noise_sigma = 0.0, use_two_samples_when_generating_quad=True): 
        self.mnist_path = mnist_path
        self.train = train
        self.use_textures = use_textures
        self.grayscale_textures = grayscale_textures
        self.texture_class_whitelist = texture_class_whitelist
        self.image_dimension = image_dimension
        self.noise_sigma = noise_sigma
        self.dtd_path = dtd_path
        self.brightness_value_background = 0.18
        self.brightness_value_foreground = 1.0 - self.brightness_value_background
        self.use_two_samples_when_generating_quad = use_two_samples_when_generating_quad

        mnist = datasets.MNIST(root=self.mnist_path, train=self.train, download=True, transform=None)

        self.mnist = self._setup_mnist_data(mnist, image_dimension, dilate=False)
        self.mnist_dilated = self._setup_mnist_data(mnist, image_dimension, dilate=True)

        self.labels = self._setup_mnist_labels(mnist)

        dtd_dataset = datasets.DTD(root=self.dtd_path, download=True, transform=None)
        self.textures = self._setup_texture_data(dtd_dataset, self.image_dimension, self.grayscale_textures, self.texture_class_whitelist)

        self.thickness = self.calculate_thickness(mnist)

        self.first_index = 0

    def set_brightness_value(self, foreground, background): 
        self.brightness_value_foreground = foreground
        self.brightness_value_background = background

    def __getitem__(self, index): 
        image = self.create_image(index)
        return [self.labels[index], torch.sum(image).item(), self.thickness[index]], image

    def get_label_names(self): 
        return ["DIGIT_CLASS", "PIXEL_SUM", "THICKNESS"]

    def __len__(self):
        return len(self.labels)
 
    def create_image(self, index, quad_index = 0):

        image = self._create_image(index, quad_index, False)
        
        should_reshape = len(image.shape) != 3

        if should_reshape: 
            image = image.reshape(1, self.image_dimension, self.image_dimension)

        image = torch.tensor(image)

        return image

    def create_image_augmented(self, index, quad_index = 0):

        image = self._create_image(index, quad_index, True)

        should_reshape = len(image.shape) != 3

        if should_reshape: 
            image = image.reshape(1, self.image_dimension, self.image_dimension)

        image = torch.tensor(image)

        return image

    def calculate_skeleton_length(self, image) -> int:
        return skeletonize(image).sum()

    def calculate_thickness(self, dataset): 

        thickness = []

        for i in range(len(dataset)):
            image = np.array(dataset[i][0])

            skeleton_length = self.calculate_skeleton_length(image)
            pixel_sum = image.sum()

            thickness.append(pixel_sum / skeleton_length)

        return thickness

    def _create_image(self, index, quad_index, rotate): 
        

        if self.use_two_samples_when_generating_quad == False:
            if quad_index == 0: 
                self.first_index = index
            
            image = self.mnist[self.first_index] if (quad_index in [0, 2]) else self.mnist_dilated[self.first_index]
        else: 
            # Hacky way for dilating image 2 and it's augmented version in the quads.
            image = self.mnist[index] if (quad_index in [0, 2]) else self.mnist_dilated[index]
        image = image.astype(np.float32)
        image /= 255.0

        if rotate: 
            image = self._augment_image_rotation(image, np.random.randint(0, 364))

        if self.use_textures:
            image = self._apply_textures_on_image(image, self.textures, self.image_dimension, self.brightness_value_foreground, self.brightness_value_background)

        return image

    def _augment_image_rotation(self, image, angle):

        # albumentations.rotate is around 10x faster than the previously used ndimage.rotate
        new_image = albumentations.rotate(image, angle=angle)

        return new_image

    def _apply_textures_on_image(self, image, textures, image_dimension, brightness_value_foreground, brightness_value_background):
        
        texture_1 = self._change_brightness_on_image(textures[np.random.randint(0, len(textures))], brightness_value_foreground)
        texture_2 = self._change_brightness_on_image(textures[np.random.randint(0, len(textures))], brightness_value_background)
        
        texture_1 = self._augment_image_rotation_crop_and_resize(texture_1, np.random.randint(0, 364), image_dimension)
        texture_2 = self._augment_image_rotation_crop_and_resize(texture_2, np.random.randint(0, 364), image_dimension)
        
        if len(texture_1.shape) == 3 and texture_1.shape[2] == 3:
            image = np.stack((image,) * 3, axis=-1)

        texture_1 = texture_1.astype(np.float32)
        texture_1 /= 255.0

        texture_2 = texture_2.astype(np.float32)
        texture_2 /= 255.0
        
        final = np.multiply(image, texture_1) + np.multiply(1.0 - image, texture_2)
        
        final = np.reshape(final, (3 if len(final.shape) == 3 else 1, image_dimension, image_dimension))
        
        return final
    
    def _augment_image_rotation_crop_and_resize(self, image, angle, image_dimension): 
        
        crop_size = self._largest_rotated_rect(image_dimension, image_dimension, np.radians(angle))

        image = self._augment_image_rotation(image, angle)

        image = self._crop_image(image, int(crop_size[1]), int(crop_size[0]))

        image = cv2.resize(image, dsize=(image_dimension, image_dimension))

        return image

    def _largest_rotated_rect(self, w, h, angle):

        quadrant = int(np.floor(angle / (np.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else np.pi - angle
        alpha = (sign_alpha % np.pi + np.pi) % np.pi

        bb_w = w * np.cos(alpha) + h * np.sin(alpha)
        bb_h = w * np.sin(alpha) + h * np.cos(alpha)

        gamma = np.arctan2(bb_w, bb_w) if (w < h) else np.arctan2(bb_w, bb_w)

        delta = np.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * np.cos(alpha)
        a = d * np.sin(alpha) / np.sin(delta)

        y = a * np.cos(gamma)
        x = y * np.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def _calculate_mean_brightness(self, image):

        is_grayscale = len(image.shape) == 2 or image.shape[2] == 1
        
        if is_grayscale: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        return np.mean(v)

    def _change_brightness(self, image, value):
        
        is_grayscale = len(image.shape) == 2 or image.shape[2] == 1
        
        if is_grayscale: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value > 0: 
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += int(value)
        else: 
            lim = np.abs(value)
            v[v <= lim] = 0
            v[v > lim] = v[v > lim] - np.array(np.abs(value), dtype=np.uint8)

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        if is_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def _change_brightness_on_image(self, image, brightness_value): 
        
        mean_brightness = self._calculate_mean_brightness(image)

        current_difference = (brightness_value * 255) - mean_brightness
        
        image = self._change_brightness(image, current_difference)

        return image

    def _setup_mnist_labels(self, mnist_dataset): 
        return [mnist_dataset[i][1] for i in range(len(mnist_dataset))]

    def _setup_texture_data(self, dtd_dataset, image_dimension, grayscale_textures, texture_class_whitelist = []):
        
        dataset = []

        texture_class_indices = [dtd_dataset.class_to_idx[current] for current in texture_class_whitelist]

        for i in range(len(dtd_dataset)):

            if not texture_class_whitelist == [] and not dtd_dataset[i][1] in texture_class_indices:
                continue

            if grayscale_textures:
                dataset.append(cv2.cvtColor(self._crop_image(np.asarray(dtd_dataset[i][0], dtype=np.uint8), image_dimension, image_dimension), cv2.COLOR_BGR2GRAY))
            else: 
                dataset.append(self._crop_image(np.asarray(dtd_dataset[i][0], dtype=np.uint8), image_dimension, image_dimension))
        
        return dataset

    def _crop_image(self, image, width, height): 
        middle_x = image.shape[0] // 2
        middle_y = image.shape[1] // 2

        return image[middle_x - (width // 2) : middle_x + (width // 2), middle_y - (height // 2) : middle_y + (height // 2)]
    
    def _setup_mnist_data(self, mnist_dataset, image_dimension, dilate): 
        
        new_dataset = []

        for i in range(len(mnist_dataset)):
            image = mnist_dataset[i][0]

            # Converting to numpy array, resizing to image_dimension x image_dimension
            image = np.asarray(image, dtype=np.uint8)
            image = cv2.resize(np.array(image, dtype=np.uint8), dsize=(image_dimension, image_dimension), interpolation=cv2.INTER_CUBIC)

            if dilate: 
                kernel = np.ones((3, 3), np.uint8)
                image = np.asarray(cv2.dilate(image, kernel, iterations=3), dtype=np.uint8)
            
            new_dataset.append(image)

        return new_dataset

class EMNIST_Dataset(MNIST_Dataset):
    def __init__(self, emnist_path, dtd_path, train, use_textures, grayscale_textures, texture_class_whitelist, image_dimension = 256, noise_sigma = 0.0, use_two_samples_when_generating_quad=True): 
        self.emnist_path = emnist_path
        self.train = train
        self.use_textures = use_textures
        self.grayscale_textures = grayscale_textures
        self.texture_class_whitelist = texture_class_whitelist
        self.image_dimension = image_dimension
        self.noise_sigma = noise_sigma
        self.dtd_path = dtd_path
        self.brightness_value_background = 0.18
        self.brightness_value_foreground = 1.0 - self.brightness_value_background
        
        emnist = datasets.EMNIST(root=self.emnist_path, split="letters", train=self.train, download=True, transform=None)

        self.mnist = self._setup_mnist_data(emnist, image_dimension, dilate=False)
        self.mnist_dilated = self._setup_mnist_data(emnist, image_dimension, dilate=True)

        self.labels = self._setup_mnist_labels(emnist)

        dtd_dataset = datasets.DTD(root=self.dtd_path, download=True, transform=None)
        self.textures = self._setup_texture_data(dtd_dataset, self.image_dimension, self.grayscale_textures, self.texture_class_whitelist)
        self.thickness = self.calculate_thickness(emnist)

        self.use_two_samples_when_generating_quad = use_two_samples_when_generating_quad

class Landscape_Rank_Rotation(Base_Dataset): 
    def __init__(self, landscape_path, two_samples=True):
        self.two_samples = two_samples
        self.landscape_path = landscape_path
        self.landscape_images_paths = os.listdir(self.landscape_path)

        # Removing all strings that does not contain .jpg 
        self.landscape_images_paths = [self.landscape_path + "/" + x for x in self.landscape_images_paths if ".jpg" in x]

        self.rotation_augmentation = transforms.RandomRotation(degrees=(0, 360))

        self.all_augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(180), # The size of the images from this dataset is 180 x 180, since sin(45) * 256 = 180
        ])

        self.augmentations = transforms.Compose([
            transforms.RandomChannelPermutation(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)
        ])

        self.q_1_temp_image = None
        self.q_2_temp_image = None
    
    def load_image(self, image_path):
        img = Image.open(image_path)
        if img.mode != "RGB": 
            img = np.array(img.convert("RGB"))
        else: 
            img = np.array(img)

        return Image.fromarray(cv2.resize(img, dsize=(256, 256)))

    def create_image(self, index, quad_index=0):

        if quad_index == 0:
            self.q_1_temp_image = self.all_augmentations(self.rotation_augmentation(self.load_image(self.landscape_images_paths[index])))
            image = self.q_1_temp_image
        elif quad_index == 1: 
            self.q_2_temp_image = self.all_augmentations(self.rotation_augmentation(self.load_image(self.landscape_images_paths[index])))
            image = self.q_2_temp_image

        return image 

    def create_image_augmented(self, index, quad_index=0):
        if quad_index == 2:
            image = self.augmentations(self.q_1_temp_image)
        else: 
            image = self.augmentations(self.q_2_temp_image)

        return image

    def get_label_names(self):
        return ["ROTATION"]

    def save_sample(self, name):
        label, image = self.__getitem__(np.random.randint(0, len(self)))
        print(image.shape)
        image = image.cpu().permute((1, 2, 0)).numpy()

        path_first_part = f"data/results_data/{name}"
        path = f"{path_first_part}/single_sample.png"

        Image.fromarray(image).save(path)


    def __getitem__(self, index):
        angle = index % 360
        image = self.load_image(self.landscape_images_paths[index])
        image = albumentations.rotate(np.array(image), angle=angle)
        image = self.all_augmentations(image)

        return [angle], image

    def __len__(self):
        return len(self.landscape_images_paths)