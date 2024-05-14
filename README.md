# Unsupervised Representation Learning through Ranking on Image Data

This GitHub Repository contains the code and resources for my Master Thesis in Computational Science at the University of Oslo, Autumn 2024.

## Installation
Python 3.9 and pip is required for running the code. Use pip3 for installing all the nescessary packages 
```
pip3 install -r requirements.txt
```

## Running an experiment
The thesis conducted multiple experiments. This required the codebase to be structed in a way for encapsulating each experiments, making it easy to run one experiment at a time. The [configs](configs/) folder holds configurations for all the experiments. Use these configuration to run experiments with
```
python3 src/main.py --config configs/experiment_x.json
```
for running experiment x. Replace x with the wanted experiment nr. This will start the training process of the experiment. For only running evaluation on pre-trained models, use can run the experiment with the `no-train` argument
```
python3 src/main.py --config configs/experiment_x.json --no-train
```
This will run the evaluation for the last saved model for the experiment. The evaluation results produced from this can be found in the [data/results_data](data/results_data/) folder.

For running evaluation on a saved model from a previous epoch, one can specify this with the `epoch` argument

```
python3 src/main.py --config configs/experiment_x.json --no-train --epoch 123
```

The `epoch` argument can also be changed for assigning the maximum amount of training epochs, before stopping.

Every experiment runs on a fixed seed, making the results in the thesis reproducible. However, the seed can also be changed with the `seed` argument to something else like
```
python3 src/main.py --config configs/experiment_x.json --seed 456
```
This will change everything from the random shuffling in the dataloaders, to the random augmentations in all datasets.


See [src/main.py](src/main.py) for a full list of all supported arguments.

## Experiments

The table below is listing where all the experiments in the thesis are located and their respective configuration files.

| Configuration           | Experiment description                             | Section in thesis |
|-------------------------|----------------------------------------------------|-------------------|
| experiment_1.json       | Ranking size of binary colored rectangles          |               9.1 |
| experiment_2_0.2.json   | Ranking size of rectangles with noise $\sigma=0.2$ |               9.2 |
| experiment_2_0.4.json   | Ranking size of rectangles with noise $\sigma=0.4$ |               9.2 |
| experiment_2_0.8.json   | Ranking size of rectangles with noise $\sigma=0.8$ |               9.2 |
| experiment_2_1.6.json   | Ranking size of rectangles with noise $\sigma=1.6$ |               9.2 |
| experiment_3.json       | Ranking color of rectangles                        |               9.3 |
| experiment_8.json       | Ranking view-angle on the 3D-cubes dataset         |                10 |
| experiment_10.json      | Ranking out-of-distribution rectangles             |              11.1 |
| experiment_11.json      | Ranking out-of-distribution rectangles             |              11.1 |
| experiment_12.json      | Ranking out-of-distribution rectangles             |              11.1 |
| experiment_14.json      | Ranking out-of-distribution rectangles             |              11.1 |
| experiment_15.json      | Ranking out-of-distribution rectangles             |              11.1 |
| experiment_16.json      | Ranking out-of-distribution 3D-cubes               |              11.1 |
| experiment_17.json      | Ranking out-of-distribution 3D-cubes               |              11.1 |
| experiment_18.json      | Ranking out-of-distribution 3D-cubes               |              11.1 |
| experiment_19.json      | Ranking out-of-distribution 3D-cubes               |              11.1 |
| experiment_20.json      | Ranking out-of-distribution 3D-cubes               |              11.1 |
| experiment_21.json      | Ranking features from (E)MNIST                     |                15 |
| experiment_22.json      | Ranking features from (E)MNIST                     |                15 |
| experiment_23.json      | Ranking features from (E)MNIST                     |                15 |
| experiment_34.json      | Ranking features from (E)MNIST                     |                15 |
| experiment_35.json      | Ranking features from (E)MNIST                     |                15 |
| experiment_36.json      | Ranking features from (E)MNIST                     |                15 |
| experiment_48.json      | Directional ranking                                |                12 |
| experiment_48_swap.json | Directional ranking                                |                12 |
| experiment_58.json      | Ranking caltech patches                            |              16.1 |
| experiment_68.json      | Ranking rotation of landscape images               |              16.2 |
| experiment_72.json      | Ranking with DiffSort                              |                13 |
| experiment_74.json      | Ranking multiple characteristics Gramian           |              14.1 |
| experiment_75.json      | Ranking multiple characteristics Spearman          |              14.2 |
| experiment_77.json      | Ranking with DiffSort                              |                13 |
| experiment_78.json      | Ranking with DiffSort                              |                13 |