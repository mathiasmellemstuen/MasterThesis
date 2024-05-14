import numpy as np
from save import save_numpy_array_to_csv

current_csv_line = {}
all_csv_lines = []

def csv_retrieve_last(key): 
    global current_csv_line
    return current_csv_line[key]

def csv_input(key, value): 
    global current_csv_line, all_csv_lines
    if key in current_csv_line.keys():
        current_csv_line[key].append(value)
    else:
        current_csv_line[key] = [value]

def csv_step(): 
    global current_csv_line, all_csv_lines
    all_csv_lines.append(current_csv_line)
    current_csv_line = {}

def csv_save(name, file_name): 
    global current_csv_line, all_csv_lines
    keys = []

    # Creating a list of all the keys 
    for line in all_csv_lines: 
        for key in line.keys(): 
            if not key in keys: 
                keys.append(key)

    # Getting the length of the array
    n = len(all_csv_lines)
    m = len(keys)

    # Creating the data array
    data = np.zeros((n, m))

    # Populating the data array
    for i, line in enumerate(all_csv_lines): 
        for key in line.keys(): 
            item = line[key]
            item = np.mean(item)
            x_index = keys.index(key)
            data[i, x_index] = item


    # Saving to csv 
    save_numpy_array_to_csv(name, file_name, keys, data)


if __name__ == "__main__": 

    csv_input("key_1", 5)
    csv_input("key_2", 2)
    csv_step()
    csv_input("key_3", 6)
    csv_input("key_2", 3)
    csv_input("key_4", 1)
    csv_input("key_4", 2)
    csv_step()
    csv_input("key_5", 89)
    csv_step()
    csv_input("key_5",89)
    csv_step()
    csv_save("test", "test")