# AirGNN

This repository contains the code for a Graph Neural Network called AirGNN, that is used for a Digital Twin.
The Graph Neural Network is implemented using PyTorch in Python.

The Graph Neural Network is trained on graph data simulated by the code in `./air_sim`.

## Installation

To run this project you will need to have a working installation of [python 3.8](https://www.python.org/downloads/release/python-380/).

It is recommended to use a [conda](https://docs.conda.io/projects/miniconda/en/latest/) environment for this.

To create a conda environment called `AirGNN` with python 3.8 use the following command:

`conda create -n AirGNN python=3.8`

To activate the `AirGNN` environment use the following command:

`conda activate AirGNN`

To install all the required python modules use the following command:

`pip install -r requirements.txt`

## Usage

### Simulation

To create a set of 100 random training appartement configurations and 20 random testing appartement configurations run the following command:

`python air_sim/generate_random_appartements.py`

The json appartement configs will be created in the directory `./appartements/json_configs`.

To then run the simulation on the appartement configs and to create the networkx graphs and labels run the following command:

`python air_sim/run_simulation.py`

The graphs will be created in `./appartements/graphs/test/nx_Graph.pickle` and `./appartements/graphs/train/nx_Graph.pickle`.

The files containing the simulation results will be created in `./appartements/graphs/test/labels.json` and `./appartements/graphs/train/labels.json`.

### Training the Graph Neural Network

To train the Graph Neural Network use the following command:

`python air_gnn/train.py`

The training parameters can be specified with command line arguments. To view the training parameters run:

`python air_gnn/train.py --help`

The best working parameters are set as the default parameters.

The trained models will be saved under `./models`. The logs of the training will be saved under `./logs`.

There are two types of log files:

- Files ending with `.csv` are files that log the `Total_Loss`, `Diff` which is the Difference to the previous `Total_Loss`, `Total_Error` and `Avg_Error` for each epoch. This will be written, when the verbosity level is at least 1.
- Files ending with `.log` are files that save the current predictions of the model compared to the labels every 100 epochs. This will be written, when the verbosity level is 2.

The training can be aborted by a KeyboardInterrupt (`Ctrl+C`) at any point. The model can be saved anyways.

#### Grid Search

To find good hyperparameters for the model use the python file `./air_gnn/grid_search.py`.

#### Retraining

To retrain a model use the following command:

`./air_gnn/retrain.py`

The path to the model to retrain will have to be provided with the parameter `-m`. The number of episodes for the retraining can be provided with the parameter `-e`, the default value is 1000.
