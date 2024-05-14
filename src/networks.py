import torch
from utils.get_device import get_device
import timm
from torchvision import models
from collections import OrderedDict

class Network_1(torch.nn.Module): 
    
    def __init__(self, input_image_channels):
        super(Network_1, self).__init__()
        self.input_image_channels = input_image_channels

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_image_channels, 8, 3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, 16, 3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(1568, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.net(x)

class Network_2_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Network_2_Block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        self.downsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=self.stride, bias=False),
        )

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding=self.padding),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride = 1),
            torch.nn.LeakyReLU(),
        )

        self.block_final_activation = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.block_final_activation(self.downsampler(x) + self.block(x))

class Network_2(torch.nn.Module): 
    def __init__(self, input_image_channels, input_linear_layers_size=2048, out_size=1):
        super(Network_2, self).__init__()

        self.input_image_channels = input_image_channels

        self.net = torch.nn.Sequential(
            Network_2_Block(in_channels=input_image_channels, out_channels=8, kernel_size=3, stride=2, padding=3),
            Network_2_Block(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=3),
            Network_2_Block(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=3),
            Network_2_Block(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=3),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(input_linear_layers_size, input_linear_layers_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(input_linear_layers_size // 2, input_linear_layers_size // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(input_linear_layers_size // 4, input_linear_layers_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(input_linear_layers_size // 2, out_size),
        )

    def forward(self, x): 
        return self.net(x)


class ConvNeXtModified(torch.nn.Module):
    def __init__(self, model_name, out_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)

        self.new_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(12288, out_classes)
        )
        self.model.head = self.new_head

    def forward(self, image_batch): 
        return self.model(image_batch)

class ConvNeXtTinyModified(ConvNeXtModified): 
    def __init__(self, out_classes = 1):
        super().__init__("convnext_tiny", out_classes=out_classes)

class ConvNeXtSmallModified(ConvNeXtModified): 
    def __init__(self, out_classes = 1):
        super().__init__("convnext_small", out_classes=out_classes)

class ConvNeXtBaseModified(ConvNeXtModified): 
    def __init__(self, out_classes = 1):
        super().__init__("convnext_base", out_classes=out_classes)

class Identity(torch.nn.Module):
    def __init__(self): 
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class TinyEncoder(torch.nn.Module):
    def __init__(self): 
        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, 16, 3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2),
            torch.nn.Flatten()
        )

    def forward(self, x): 
        return self.model(x)

class Tiny_Projection_Head(torch.nn.Module):
    def __init__(self, num_in_features=512, out_features=1): 
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_in_features, out_features)
        )

    def forward(self, image_batch): 
        return self.net(image_batch)
    
def save_model(model, path): 
    torch.save(model.state_dict(), path) 

def load_model(model, path): 
    model.load_state_dict(torch.load(path, map_location=torch.device(get_device())))