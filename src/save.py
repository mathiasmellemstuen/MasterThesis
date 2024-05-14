from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt

def save_numpy_array_to_csv(name, file_name, column_names, array):
    save_numpy_array_to_csv_full_path(f"data/results_data/{name}/{file_name}.csv", column_names, array)

def save_numpy_array_to_csv_full_path(full_path, column_names, array):

    with open(full_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Writing the column names first
        writer.writerow(column_names) 
        
        # Then writing all data
        writer.writerows(array)

def save_scores_and_parameters_to_csv(name, file_name, scores_and_labels):

    column_names = ["SCORE"] + [label.label_type.name for label in scores_and_labels[0][0]]
    
    samples_length = len(scores_and_labels)
    labels_length = len(scores_and_labels[0][0])

    csv_data = np.zeros((samples_length, 1 + labels_length))
    
    for i in range(samples_length):
        csv_data[i, 0] = scores_and_labels[i][1]
    
    for i in range(labels_length):
        csv_data[:, i + 1] = [x[0][i].label for x in scores_and_labels]

    with open(f"data/results_data/{name}/{file_name}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Writing the column names first
        writer.writerow(column_names) 

        # Then writing all data
        writer.writerows(csv_data)