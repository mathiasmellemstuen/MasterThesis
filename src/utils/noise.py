import torch

def noise_image(image_tensor, sigma):

    if sigma == 0.0: 
        return image_tensor
    
    noise = create_noise(sigma, image_tensor.shape)

    return apply_noise_image_to_image(image_tensor, noise)

def create_noise(sigma, shape): 
    return (2.0 * torch.rand(shape) - 1.0) * sigma

def apply_noise_image_to_image(image_tensor, noise_image_tensor): 

    noised_image = (image_tensor / image_tensor.max()) + noise_image_tensor 

    return torch.clamp(noised_image, 0.0, 1.0)