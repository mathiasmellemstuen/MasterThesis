import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from dataset import QuadDataset
from quad_network import QuadNetwork
from numba import jit 
from os import system

@jit(nopython=True)
def num_bubble_sort_swaps(arr, ascending_order): 

    n = len(arr)
    num_swaps = 0

    step = 1

    if ascending_order:
        outer_start = 0
        outer_end = n - 1
        inner_start = 0
        inner_end = n - 1
    else: 
        outer_start = n - 1
        outer_end = 1
        inner_start = n - 1
        inner_end = 1
        step = -1

    for i in range(outer_start, outer_end, step): 
        print(f"Iteration: {i} / {n}")

        for j in range(inner_start, inner_end, step): 
            
            # Swapping in this case
            if arr[j] > arr[j + step]: 
                arr[j], arr[j + step] = arr[j + step], arr[j]
                num_swaps += 1

    return num_swaps

def sorting_error(name : str, model : QuadNetwork, dataset : QuadDataset, scores_index=2, true_values_index=0, postfix=""):
    
    data = np.genfromtxt(f"data/results_data/{name}/quad_labels_and_prediction_scores.csv", dtype=float, delimiter=",", skip_header=1)
    ranking_scores = data[:, scores_index]
    true_values = data[:, true_values_index]

    ranking_values_sorting_ind = np.argsort(ranking_scores)
    arr = true_values[ranking_values_sorting_ind]

    num_swaps_increasing = num_bubble_sort_swaps(np.copy(arr), ascending_order=True)
    num_swaps_decreasing = num_bubble_sort_swaps(np.copy(arr), ascending_order=False)

    print(num_swaps_increasing, num_swaps_decreasing)

    # Testing for both direction and choosing for the smallest error. 
    num_swaps = num_swaps_increasing if num_swaps_decreasing >= num_swaps_increasing else num_swaps_decreasing

    lines = [str(num_swaps) + "\n", str(num_swaps / (len(arr) * len(arr)))]
    with open(f"data/results_data/{name}/sorting_error{postfix}.txt", "w") as f: 
        f.writelines(lines)

def spearman_correlation(name : str, model : QuadNetwork, dataset : QuadDataset, scores_index = 2, true_values_index=0, postfix=""): 
    data = np.genfromtxt(f"data/results_data/{name}/quad_labels_and_prediction_scores.csv", dtype=float, delimiter=",", skip_header=1)
    ranking_scores = data[:, scores_index]
    true_values = data[:, true_values_index]

    res = stats.spearmanr(true_values, ranking_scores)

    lines = [str(res.statistic)]
    with open(f"data/results_data/{name}/spearman_correlation{postfix}.txt", "w") as f: 
        f.writelines(lines)

if __name__ == "__main__":
    sorting_error("new/exp_75_rectangles_ranking_size_binary_nd_quadnetwork_spearman", None, None, scores_index=7, postfix="_score_2_size")
