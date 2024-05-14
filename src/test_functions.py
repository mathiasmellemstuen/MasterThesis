from quad_network import QuadNetwork
from dataset import QuadDataset
from save import save_numpy_array_to_csv
from tqdm import tqdm
from utils.noise import *
from PIL import Image
from palettable.colorbrewer.diverging import BrBG_10
from utils.color_gradient import color_gradient_min_red_max_blue
from pathlib import Path
from fast_slic import Slic
from sklearn.linear_model import LinearRegression
from numba import jit
from save import *
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 
import matplotlib
import random

def get_quad_prediction_scores_gap(name : str, model : QuadNetwork, dataset : QuadDataset, dataset_passes : int, save : bool = True): 
    label_names = dataset.base_dataset.get_label_names()
    label_names.append("INCLUDED")
    label_names.append("SCORE")
    label_names.append("NORMALIZED_SCORE")
    
    total_length = dataset_passes * (dataset.base_dataset.included_length + dataset.base_dataset.excluded_length)

    data = np.zeros((total_length, len(label_names)))

    with tqdm(total=total_length) as progress_bar:
        progress_bar.set_description("Testing: get_quad_prediction_scores_gap")
        for i in range(dataset_passes):
            
            # Running for the included samples 
            for mode in range(0, 2): 

                dataset.base_dataset.set_mode(mode)

                for j in range(len(dataset.base_dataset)):

                    index = i * (dataset.base_dataset.included_length + dataset.base_dataset.excluded_length) + j + (mode * dataset.base_dataset.excluded_length)
                    labels, image = dataset.base_dataset[j]

                    if mode == 0:
                        print(labels)
                
                    image = image.unsqueeze(1)
                    image = image.reshape((image.shape[1], image.shape[0], image.shape[2], image.shape[3]))
                    
                    data[index, -2] = model.predict(image)
                    data[index, -3] = mode

                    for k, label in enumerate(labels):
                        data[index, k] = label

                    image = image.cpu().numpy()
                    image = image * 255
                    image = image.astype(np.uint8)
                    image = image.reshape((image.shape[3], image.shape[2], image.shape[1]))

                    progress_bar.update(1)

    max_score = np.max(data[:, len(label_names) - 2])
    min_score = np.min(data[:, len(label_names) - 2])

    for i in range(total_length):
        score = data[:, -2]
        data[:, -1] = (score - min_score) / (max_score - min_score)

    if save:
        save_numpy_array_to_csv(name, "quad_labels_and_prediction_scores", label_names, data)

    return data

def get_quad_prediction_scores(name : str, model : QuadNetwork, dataset : QuadDataset, dataset_passes: int, save: bool = True):

    label_names = dataset.base_dataset.get_label_names()
    label_names.append("IMAGE_LIGHTNESS")
    label_names.append("SCORE")
    label_names.append("NORMALIZED_SCORE")

    data = np.zeros((dataset_passes * len(dataset.base_dataset), len(label_names)))

    with tqdm(total=dataset_passes * len(dataset.base_dataset)) as progress_bar:
        progress_bar.set_description("Testing: get_quad_prediction_scores")
        for i in range(dataset_passes):
            for j in range(len(dataset.base_dataset)):

                index = i * len(dataset.base_dataset) + j
                labels, image = dataset.base_dataset[j]
            
                image = image.unsqueeze(1)
                image = image.reshape((image.shape[1], image.shape[0], image.shape[2], image.shape[3]))
                
                data[index, -2] = model.predict(image)

                for k, label in enumerate(labels):
                    data[index, k] = label
                
                image = image.cpu().numpy()
                image = image * 255
                image = image.astype(np.uint8)
                image = image.reshape((image.shape[3], image.shape[2], image.shape[1]))
                # L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
                # data[index, -3] = np.mean(L / np.max(L))

                progress_bar.update(1)

    max_score = np.max(data[:, len(label_names) - 2])
    min_score = np.min(data[:, len(label_names) - 2])

    for i in range(dataset_passes * len(dataset.base_dataset)):
        score = data[:, -2]
        data[:, -1] = (score - min_score) / (max_score - min_score)

    if save:
        save_numpy_array_to_csv(name, "quad_labels_and_prediction_scores", label_names, data)

    return data

def lime_evaluate(name : str, model : QuadNetwork, dataset : QuadDataset, number_of_samples_to_evaluate = 10): 
    data = get_quad_prediction_scores(name, model, dataset, 1, False)

    num_segments = 40
    num_samples = 1024
    p = 0.25

    images = []

    max_score = np.max(data[:, -2])
    min_score = np.min(data[:, -2])

    print(data[:, -1])
    print(min_score, max_score)
    
    all_scores = np.linspace(min_score, max_score, number_of_samples_to_evaluate)

    all_images = []
    all_segments = []
    all_heatmaps = []
    all_selected_scores = []

    for i in range(number_of_samples_to_evaluate):
        print(f"Want to find a value close to {all_scores[i]}")

        index = (np.abs(data[:, -2] - all_scores[i])).argmin()

        all_selected_scores.append(data[index, -2])

        labels, image = dataset.base_dataset[index]

        image = image.cpu().numpy()
        image = image * 255
        image = image.astype(np.uint8)
        image = image.reshape((image.shape[1], image.shape[2], image.shape[0]))

        noise_image = torch.rand(image.shape) * 255
        noise_image = noise_image.numpy()

        slic = Slic(num_components=num_segments)
        segments = slic.iterate(image)


        X = np.zeros((num_segments, num_samples))
        y = np.zeros(num_samples)

        for i in range(num_segments):
            for j in range(num_samples): 
                X[i, j] = 1 if np.random.uniform(0.0, 1.0) >= p else 0
    
        for i in range(num_samples): 
            current_image = image.copy().astype(np.float32)

            x = X[:, i]

            for j, value in enumerate(x):
                current_image[segments == j] = noise_image[segments == j] if value == 1 else current_image[segments == j]

            current_image = current_image.reshape((3, 128, 128))
            current_image = current_image.astype(np.float32)
            current_image /= 255.0
            current_image = torch.from_numpy(current_image)
            current_image = current_image.unsqueeze(dim=0)

            y[i] = model.predict(current_image).item()
        

        lr = LinearRegression()
        linear_model = lr.fit(X.T, y)

        beta = linear_model.coef_

        evaluation_heatmap = np.zeros((image.shape[0], image.shape[1]))
        evaluation_heatmap = beta[segments]
        
        all_images.append(image)
        all_segments.append(segments)
        all_heatmaps.append(evaluation_heatmap)

    for i in range(len(all_images)): 
        score_folder_name = "neg" + str(np.abs(all_selected_scores[i])) if all_selected_scores[i] < 0 else str(all_selected_scores[i])
        Path(f"data/results_data/{name}/lime_evaluation/{score_folder_name}").mkdir(parents=True, exist_ok=True)
        matplotlib.image.imsave(f"data/results_data/{name}/lime_evaluation/{score_folder_name}/image.png", all_images[i])
        matplotlib.image.imsave(f"data/results_data/{name}/lime_evaluation/{score_folder_name}/segments.png", all_segments[i])
        matplotlib.image.imsave(f"data/results_data/{name}/lime_evaluation/{score_folder_name}/evaluation_heatmap.png", all_heatmaps[i])

def get_samples_from_score_bins(name : str, model : QuadNetwork, dataset : QuadDataset, amount_of_bins = 10, amount_of_samples_per_bin = 10):
    data = get_quad_prediction_scores(name, model, dataset, 1, False)


    min_score = np.min(data[:, -2])
    max_score = np.max(data[:, -2])

    bin_width = (max_score - min_score) / amount_of_bins

    print(min_score, max_score)

    for i in range(1, amount_of_bins + 1):
        samples_in_bin = []
        scores_in_bin = []

        bin_min = (i - 1) * bin_width
        bin_max = i * bin_width

        indices = list(range(len(dataset.base_dataset)))
        random.shuffle(indices)

        for j in indices:
            if bin_min <= data[j, -2] < bin_max:
                labels, image = dataset.base_dataset[j]
                image = image.unsqueeze(1)
                image = image.reshape((image.shape[1], image.shape[0], image.shape[2], image.shape[3]))
                image = image.numpy().reshape(image.shape[2], image.shape[3], image.shape[1])
                samples_in_bin.append(image)
                scores_in_bin.append([data[j, -2]])

                if len(samples_in_bin) >= amount_of_samples_per_bin:
                    break
        
        folder_name = f"samples_in_score_bins/bin_{i}/"
        Path(f"data/results_data/{name}/{folder_name}").mkdir(parents=True, exist_ok=True)

        save_numpy_array_to_csv(name, f"{folder_name}/scores", ["SCORE"], np.array(scores_in_bin))

        for i, sample in enumerate(samples_in_bin): 
            plt.figure()
            plt.imshow(sample)
            plt.gca().set_axis_off()
            plt.savefig(f"data/results_data/{name}/{folder_name}/{i}.png", bbox_inches="tight", pad_inches=0)
            plt.close()

def create_before_after_noise_images(name : str, model : QuadNetwork, dataset : QuadDataset, image_path : str, noise_sigma : float):
    image = torch.tensor(np.array(Image.open(image_path), dtype=np.uint8))

    # Saving the before noise image
    Image.open(image_path).save(f"data/results_data/{name}/before_noise.png")

    # Noise image
    noise_image = create_noise(noise_sigma, image.shape)

    plt.figure()
    plt.imshow(noise_image, cmap=BrBG_10.mpl_colormap, interpolation="none")
    plt.gca().set_axis_off()
    plt.savefig(f"data/results_data/{name}/noise.png", bbox_inches="tight", pad_inches=0)

    noised_image = apply_noise_image_to_image(image, noise_image)

    to_pil = T.ToPILImage()
    noise_image_pil = to_pil(noised_image)
    noise_image_pil.save(f"data/results_data/{name}/after_noise.png")

def create_color_gradient_image(name: str, model : QuadNetwork, dataset : QuadDataset):
    image = np.zeros((100, 300, 3), dtype=np.uint8)

    # Constructing a image from the gradient
    for i in range(300):
        color = color_gradient_min_red_max_blue(0, 300, i)
        image[:, i, 0] = color[0]
        image[:, i, 1] = color[1]
        image[:, i, 2] = color[2]

    # Saving the gradient to file
    matplotlib.image.imsave(f"data/results_data/{name}/r_to_b_gradient.png", image)

def get_nd_quad_predictions(name : str, model : QuadNetwork, dataset : QuadDataset, n : int,  dataset_passes : int): 
    label_names = dataset.base_dataset.get_label_names()

    for i in range(n): 
        label_names.append(f"SCORE_{i + 1}")
    
    for i in range(n): 
        label_names.append(f"NORMALIZED_SCORE_{i + 1}")

    data = np.zeros((dataset_passes * len(dataset.base_dataset), len(label_names)))
    
    label_count = 0

    with tqdm(total=dataset_passes * len(dataset.base_dataset)) as progress_bar:
        progress_bar.set_description("Testing: get_quad_prediction_scores")
        for i in range(dataset_passes):
            for j in range(len(dataset.base_dataset)):

                index = i * len(dataset.base_dataset) + j
                labels, image = dataset.base_dataset[j]
            
                image = image.unsqueeze(1)
                image = image.reshape((image.shape[1], image.shape[0], image.shape[2], image.shape[3]))
                

                for k, label in enumerate(labels):
                    data[index, k] = label
                
                label_count = len(labels)

                d = model.predict(image)

                for k in range(n): 
                    data[index, k + label_count] = d[k]

                progress_bar.update(1)

    save_numpy_array_to_csv(name, "quad_labels_and_prediction_scores", label_names, data)

    return data


def data_rasterize_lossy(name : str, model : QuadNetwork, dataset : QuadDataset, alpha, postfix = "", scores_index=2, true_values_index=0): 

    from_path = f"data/results_data/{name}/quad_labels_and_prediction_scores.csv"
    to_path = f"data/results_data/{name}/rasterized_quad_labels_and_prediction_scores{postfix}.csv"

    header = None
    with open(from_path) as f:
        header = f.readline().strip().split(",")

    data = np.genfromtxt(from_path, dtype=float, delimiter=",", skip_header=1)

    true_values = data[:, true_values_index]
    scores = data[:, scores_index]

    @jit(nopython=True)
    def reduce_loop(true_values, scores): 

        should_include = np.zeros((data.shape[0]), dtype=np.uint)
        excluded = np.zeros((data.shape[0]))

        n = data.shape[0]

        for i in range(n): 
            for j in range(n): 
                
                if i == j: 
                    continue

                if excluded[i] == 1 or excluded[j] == 1:
                    continue

                x_1 = true_values[i] * 1000
                y_1 = scores[i]
                x_2 = true_values[j] * 1000
                y_2 = scores[j]

                distance = dist(x_1, y_1, x_2, y_2)

                if distance < alpha: 
                    should_include[i] = 1
                    excluded[j] = 1

            print(f"Iteration {i} / {n}")

        return should_include
    
    should_include = reduce_loop(true_values, scores)

    save_numpy_array_to_csv_full_path(to_path, header, data[should_include == 1])
    
@jit(nopython=True)
def dist(x_1, y_1, x_2, y_2): 
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) **2)