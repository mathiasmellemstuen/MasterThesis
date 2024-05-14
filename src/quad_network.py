import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.mps
import torch_optimizer as optim
import pandas as pd
import diffsort
from utils.get_device import get_device
from torchsummary import summary
from csv_saver import *

class QuadNetwork(torch.nn.Module):
    def __init__(self, network):
        super(QuadNetwork, self).__init__()

        self.network = network

    def forward(self, images_batch):
        
        score_image_1 = self.network(images_batch[:, :, :, :, 0])
        score_image_2 = self.network(images_batch[:, :, :, :, 1])
        score_image_3 = self.network(images_batch[:, :, :, :, 2])
        score_image_4 = self.network(images_batch[:, :, :, :, 3])

        s = (score_image_1 - score_image_2) * (score_image_3 - score_image_4)

        loss = torch.max(torch.zeros_like(s), 1.0 - s)

        return loss.sum()

    def predict(self, image):
        image = image.to(get_device())

        return self.network(image)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path): 
        self.load_state_dict(torch.load(path, map_location=torch.device(get_device())))
        self.eval()

class DiffSortSupervisedSum(torch.nn.Module):
    def __init__(self, network):
        super(DiffSortSupervisedSum, self).__init__()

        self.network = network

        self.sorter = diffsort.DiffSortNet(sorting_network_type="bitonic", distribution="optimal", steepness=10, size=256, device=get_device())

    def forward(self, images_batch):
        
        score_image_1 = self.network(images_batch[:, :, :, :, 0])

        sorter_input = score_image_1.squeeze().unsqueeze(0)

        _, diff_sort_mat = self.sorter(sorter_input)

        targets = torch.sqrt(torch.sum(images_batch[:, :, :, :, 0], dim=(1, 2, 3))).unsqueeze(0)

        perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(targets, dim=-1)).transpose(-2, -1).float()

        loss = torch.nn.BCELoss()(diff_sort_mat, perm_ground_truth)

        return loss * images_batch.shape[0]


    def predict(self, image):
        image = image.to(get_device())

        return self.network(image)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path): 
        self.load_state_dict(torch.load(path, map_location=torch.device(get_device())))
        self.eval()

class DiffSortSelfSupervised(torch.nn.Module):
    def __init__(self, network):
        super(DiffSortSelfSupervised, self).__init__()

        self.network = network

        self.sorter = diffsort.DiffSortNet(sorting_network_type="bitonic", distribution="optimal", steepness=10, size=4, device=get_device())

    def forward(self, images_batch):
        
        score_image_1 = self.network(images_batch[:, :, :, :, 0])
        score_image_2 = self.network(images_batch[:, :, :, :, 1])

        # This should be the minimum score
        score_beta_image = self.network(images_batch[:, :, :, :, 4])

        # This should be the maximum score
        score_gamma_image = self.network(images_batch[:, :, :, :, 5])

        # 256 x 4

        diff_sort_input = torch.cat((score_beta_image, score_image_1, score_image_2, score_gamma_image), dim=-1)

        _, sort_a = self.sorter(diff_sort_input)

        Q_a = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        device=get_device()).unsqueeze(0).repeat(images_batch.shape[0], 1, 1)
        
        Q_b = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        device=get_device()).unsqueeze(0).repeat(images_batch.shape[0], 1, 1)
        
        loss_a = torch.nn.BCELoss()(sort_a, Q_a)

        loss_b = torch.nn.BCELoss()(sort_a, Q_b)

        return loss_a * loss_b * images_batch.shape[0]

    def predict(self, image):
        image = image.to(get_device())

        return self.network(image)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path): 
        self.load_state_dict(torch.load(path, map_location=torch.device(get_device())))
        self.eval()


class DiffSortSelfSupervisedv2(torch.nn.Module):
    def __init__(self, network):
        super(DiffSortSelfSupervisedv2, self).__init__()

        self.network = network

        self.sorter = diffsort.DiffSortNet(sorting_network_type="bitonic", distribution="optimal", steepness=10, size=6, device=get_device())

    def forward(self, images_batch):
        
        score_image_1 = self.network(images_batch[:, :, :, :, 0])
        score_image_2 = self.network(images_batch[:, :, :, :, 1])
        score_image_3 = self.network(images_batch[:, :, :, :, 2])
        score_image_4 = self.network(images_batch[:, :, :, :, 3])

        # This should be the minimum score
        score_beta_image = self.network(images_batch[:, :, :, :, 4])

        # This should be the maximum score
        score_gamma_image = self.network(images_batch[:, :, :, :, 5])

        # 256 x 4

        diff_sort_input = torch.cat((score_beta_image, score_image_1, score_image_2, score_image_3, score_image_4, score_gamma_image), dim=-1)

        _, sort_a = self.sorter(diff_sort_input)

        Q_a = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        device=get_device()).unsqueeze(0).repeat(images_batch.shape[0], 1, 1)
        
        Q_b = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        device=get_device()).unsqueeze(0).repeat(images_batch.shape[0], 1, 1)

        loss_a = torch.nn.BCELoss()(sort_a, Q_a)

        loss_b = torch.nn.BCELoss()(sort_a, Q_b)

        return loss_a * loss_b * images_batch.shape[0]


    def predict(self, image):
        image = image.to(get_device())

        return self.network(image)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path): 
        self.load_state_dict(torch.load(path, map_location=torch.device(get_device())))
        self.eval()

class NDQuadNetworkGram(torch.nn.Module): 
    def __init__(self, encoder, projection_heads = [], number_of_divisions=32):
        super(NDQuadNetworkGram, self).__init__()

        self.encoder = encoder
        self.projection_heads = torch.nn.ModuleList([m for m in projection_heads])
        self.n = len(self.projection_heads)
        self.number_of_divisions = number_of_divisions

    def to_gram(self, input): 

        # We reshape the input to have an extra dimension with size number_of_divisions
        input = input.reshape((input.shape[0], input.shape[1] // self.number_of_divisions, self.number_of_divisions))
        
        g_loss = 0

        # Calculating the gram matrix per element in the batch
        gram = torch.zeros((input.shape[0], self.number_of_divisions, self.number_of_divisions), device=get_device())
        for i in range(input.shape[0]):
            gram[i, :, :] = torch.matmul(input[i, :, :].T, input[i, :, :])

            g_loss += torch.sum(torch.square(torch.triu(gram[i, :, :], diagonal=1)))

        # Dividing the gram matrices in n parts
        gram = gram.reshape((input.shape[0], self.n, gram.shape[1] // self.n, gram.shape[2]))

        return gram, g_loss

    def forward(self, images_batch): 
        

        gram_1, g_1_loss = self.to_gram(self.encoder(images_batch[:, :, :, :, 0]))
        gram_2, g_2_loss = self.to_gram(self.encoder(images_batch[:, :, :, :, 1]))
        gram_3, g_3_loss = self.to_gram(self.encoder(images_batch[:, :, :, :, 2]))
        gram_4, g_4_loss = self.to_gram(self.encoder(images_batch[:, :, :, :, 3]))
        
        gram_loss = g_1_loss + g_2_loss + g_3_loss + g_4_loss
        quad_loss = torch.zeros((images_batch.shape[0], self.n))

        for i in range(self.n):
            
            projection_1 = self.projection_heads[i](gram_1[:, i, :, :])
            projection_2 = self.projection_heads[i](gram_2[:, i, :, :])
            projection_3 = self.projection_heads[i](gram_3[:, i, :, :])
            projection_4 = self.projection_heads[i](gram_4[:, i, :, :])
            
            quad_comparison = (projection_1 - projection_2) * (projection_3 - projection_4)

            # Hinge loss 
            quad_loss[:, i] = torch.squeeze(torch.max(torch.zeros_like(quad_comparison), 1.0 - quad_comparison))
        
        csv_input("LOSS_QUAD_0", torch.sum(quad_loss[:, 0]).item())
        csv_input("LOSS_QUAD_1", torch.sum(quad_loss[:, 1]).item())
        csv_input("LOSS_GRAM_LOSS", gram_loss.item())

        return torch.sum(quad_loss[:, 0]) + torch.sum(quad_loss[:, 1]) + gram_loss

    def predict(self, image):
        image = image.to(get_device())

        out_values = torch.zeros(self.n)

        gram, _ = self.to_gram(self.encoder(image))

        for i in range(self.n): 
            out_values[i] = torch.mean(self.projection_heads[i](gram[:, i, :, :]))

        return out_values

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path): 
        self.load_state_dict(torch.load(path, map_location=torch.device(get_device())))
        self.eval()

class NDQuadNetworkSpearman(torch.nn.Module): 
    def __init__(self, _, ensemble_models = []):
        super(NDQuadNetworkSpearman, self).__init__()

        self.n = len(ensemble_models)
        self.ensemble_models = torch.nn.ModuleList([m for m in ensemble_models])

    def get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(len(x))
        return ranks

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor):
        """Compute correlation between 2 1-D vectors
        Args:
            x: Shape (N, )
            y: Shape (N, )
        """
        x_rank = self.get_ranks(x)
        y_rank = self.get_ranks(y)

        n = x.size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n ** 2 - 1.0)
        return 1.0 - (upper / down)
    def forward(self, images_batch): 

        quad_loss = torch.zeros((images_batch.shape[0], self.n))

        for i in range(self.n):
            
            projection_1 = self.ensemble_models[i](images_batch[:, :, :, :, 0])
            projection_2 = self.ensemble_models[i](images_batch[:, :, :, :, 1])
            projection_3 = self.ensemble_models[i](images_batch[:, :, :, :, 2])
            projection_4 = self.ensemble_models[i](images_batch[:, :, :, :, 3])
            
            quad_comparison = (projection_1 - projection_2) * (projection_3 - projection_4)
            
            quad_loss[:, i] = torch.squeeze(torch.max(torch.zeros_like(quad_comparison), 1.0 - quad_comparison))


        spearman_loss = self.spearman_correlation(quad_loss[:, 0], quad_loss[:, 1])

        csv_input("LOSS_QUAD_0", torch.sum(quad_loss[:, 0]).item())
        csv_input("LOSS_QUAD_1", torch.sum(quad_loss[:, 1]).item())
        csv_input("LOSS_SPEARMAN", spearman_loss.item() * 512.0)

        return torch.sum(quad_loss[:, 0]) + torch.sum(quad_loss[:, 1]) + 512.0 * spearman_loss

    def predict(self, image):
        image = image.to(get_device())

        out_values = torch.zeros(self.n)

        for i in range(self.n): 
            out_values[i] = self.ensemble_models[i](image)

        return out_values

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path): 
        self.load_state_dict(torch.load(path, map_location=torch.device(get_device())))
        self.eval()

class DirectionalQuadNetwork(QuadNetwork):
    def __init__(self, network, swap_beta_gamma=False):
        super(DirectionalQuadNetwork, self).__init__(network)
        self.swap_beta_gamma=swap_beta_gamma

    def forward(self, images_batch):
        score_image_1 = self.network(images_batch[:, :, :, :, 0])
        score_image_2 = self.network(images_batch[:, :, :, :, 1])
        score_image_3 = self.network(images_batch[:, :, :, :, 2])
        score_image_4 = self.network(images_batch[:, :, :, :, 3])

        s = (score_image_1 - score_image_2) * (score_image_3 - score_image_4)
        quad_loss = torch.max(torch.zeros_like(s), 1.0 - s)

        # This should be the minimum score
        score_beta_image = self.network(images_batch[:, :, :, :, 4])

        # This should be the maximum score
        score_gamma_image = self.network(images_batch[:, :, :, :, 5])

        if self.swap_beta_gamma:
            score_beta_image, score_gamma_image = score_gamma_image, score_beta_image


        zero_tensor = torch.zeros_like(s)
        loss = \
          torch.max(zero_tensor, score_beta_image - score_image_1) \
        + torch.max(zero_tensor, score_beta_image - score_image_2) \
        + torch.max(zero_tensor, score_image_1 - score_gamma_image)  \
        + torch.max(zero_tensor, score_image_2 - score_gamma_image)  \
        + torch.max(zero_tensor, score_beta_image - score_image_3) \
        + torch.max(zero_tensor, score_beta_image - score_image_4) \
        + torch.max(zero_tensor, score_image_3 - score_gamma_image) \
        + torch.max(zero_tensor, score_image_4 - score_gamma_image)

        return loss.sum() + quad_loss.sum()