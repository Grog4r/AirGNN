import io
import json
import os
import pickle
from datetime import datetime
from time import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR


def load_graph(filename):
    return pickle.load(open(filename, "rb"))


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    else:
        print("weights_init() did nothing.")


class MessageFunction(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation="relu",
        init_weights=False,
    ):
        super(MessageFunction, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        if init_weights:
            weights_init(self.hidden_layer)
            weights_init(self.output_layer)
        self.activation = nn.ReLU()
        if activation == "selu":
            self.activation = nn.SELU()

    def forward(self, hidden_state):
        # Apply a linear transformation to the hidden state
        # This is for backwards compatibility with older model versions
        try:
            hidden_output = self.activation(self.hidden_layer(hidden_state))
        except AttributeError:
            self.activation = nn.ReLU()
            hidden_output = self.activation(self.hidden_layer(hidden_state))

        # Apply another linear transformation to get the message
        message = self.output_layer(hidden_output)

        return message


class NoMessageFunction(nn.Module):
    def forward(self, hidden_state):
        return hidden_state


def aggregation_function(graph, node, messages, verbose=False):
    aggregated_message = None
    neighbors = nx.neighbors(graph, node)
    c = 0
    for neighbor in neighbors:
        c += 1
        if aggregated_message is None:
            aggregated_message = messages[neighbor]
        else:
            aggregated_message += messages[neighbor]
    if verbose:
        print(f"Aggregation for {node}: {aggregated_message} ({c} neighbors)")
    if c == 0:
        return None
    return aggregated_message / c


class UpdateFunction(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation="relu",
        init_weights=False,
    ):
        super(UpdateFunction, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        if init_weights:
            weights_init(self.hidden_layer)
            weights_init(self.output_layer)
        self.activation = nn.ReLU()
        if activation == "selu":
            self.activation = nn.SELU()

    def forward(self, hidden_state, aggregated_message):
        # Apply a linear transformation to the hidden state
        inputs = torch.cat((hidden_state, aggregated_message))
        # This is for backwards compatibility with older model versions
        try:
            hidden_output = self.activation(self.hidden_layer(inputs))
        except AttributeError:
            self.activation = nn.ReLU()
            hidden_output = self.activation(self.hidden_layer(inputs))

        # Apply another linear transformation to get the message
        message = self.output_layer(hidden_output)

        return message


def message_passing_iteration(
    graph,
    hidden_states,
    message_function,
    aggregation_function,
    update_function,
    verbose=False,
):
    new_hidden_states = {}

    messages = {}

    c = 0
    for node in graph.nodes():
        c += 1
        # Calculate message using the message function
        messages[node] = message_function(hidden_states[node])
        if verbose:
            print(f"Message for {node}: {hidden_states[node]} -> {messages[node]}")
    if verbose:
        print(f"Calculated messages for {c} nodes.")

    c = 0
    for node in graph.nodes():
        c += 1
        # Send the message to all neighbors and aggregate
        aggregated_message = aggregation_function(graph, node, messages)
        if aggregated_message is not None:
            # Update the hidden state using the update function
            new_hidden_states[node] = update_function(
                hidden_states[node], aggregated_message
            )
        else:
            new_hidden_states[node] = hidden_states[node]

    if verbose:
        print(f"Calculated new hidden states for {c} nodes.")

    return new_hidden_states


class ReadoutFunction(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation="relu",
        init_weights=False,
    ):
        super(ReadoutFunction, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        if init_weights:
            weights_init(self.hidden_layer)
            weights_init(self.output_layer)
        self.activation = nn.ReLU()
        if activation == "selu":
            self.activation = nn.SELU()

    def forward(self, final_hidden_states):
        # Apply a linear transformation to the final hidden states
        try:
            hidden_output = self.activation(self.hidden_layer(final_hidden_states))
        # This is for backwards compatibility with older model versions
        except AttributeError:
            self.activation = nn.ReLU()
            hidden_output = self.activation(self.hidden_layer(final_hidden_states))

        # Apply another linear transformation to get the regression value
        regression_value = self.output_layer(hidden_output)

        return regression_value


def readout_module(graph, final_hidden_states, readout_function):
    outputs = {}
    for node in graph.nodes():
        outputs[node] = readout_function(final_hidden_states[node])
    return outputs


def no_readout_module(graph, final_hidden_states, _):
    outputs = {}
    for node in graph.nodes():
        outputs[node] = final_hidden_states[node].item()[0]


class GNN_Model:
    def __init__(
        self,
        train_graph: nx.Graph,
        train_labels: dict,
        test_graph: nx.Graph,
        test_labels: dict,
        hidden_message: int = 16,
        hidden_update: int = 32,
        hidden_readout: int = 16,
        n_iterations: int = 5,
        optim_learning_rate: float = 0.005,
        optim_weight_decay: float = 0,
        model_name: str = "model",
        activation: str = "relu",
        init_weights=False,
        criterion: str = "MSE",
        decay: bool = False,
        no_message_function: bool = False,
        no_readout_function: bool = False,
    ):
        # self.device = (
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if torch.backends.mps.is_available()
        #     else "cpu"
        # )
        self.device = "cpu"
        print(f"The device is: {self.device}")

        # torch.manual_seed(42)
        # np.random.seed(42)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(42)

        self.train_graph = train_graph
        self.test_graph = test_graph

        self.train_label_tensors = {}
        for node in self.train_graph.nodes():
            self.train_label_tensors[node] = torch.tensor(
                train_labels[node], dtype=torch.float32
            ).to(self.device)

        self.test_label_tensors = {}
        for node in self.test_graph.nodes():
            self.test_label_tensors[node] = torch.tensor(
                test_labels[node], dtype=torch.float32
            ).to(self.device)

        self.n_features = self.show_graph_info(self.train_graph)

        if no_message_function:
            self.message_function = NoMessageFunction().to(self.device)
        else:
            self.message_function = MessageFunction(
                input_size=self.n_features,
                hidden_size=hidden_message,
                output_size=self.n_features,
                activation=activation,
                init_weights=init_weights,
            ).to(self.device)

        self.update_function = UpdateFunction(
            input_size=2 * self.n_features,
            hidden_size=hidden_update,
            output_size=self.n_features,
            activation=activation,
            init_weights=init_weights,
        ).to(self.device)

        if not no_readout_function:
            self.readout_function = ReadoutFunction(
                input_size=self.n_features,
                hidden_size=hidden_readout,
                output_size=1,
                activation=activation,
                init_weights=init_weights,
            ).to(self.device)
            self.readout_module = readout_module
        else:
            self.readout_function = None
            self.readout_module = no_readout_module

        self.parameters = list(self.update_function.parameters())

        if not no_message_function:
            self.parameters += list(self.message_function.parameters())
        if not no_readout_function:
            self.parameters += list(self.readout_function.parameters())

        self.criterion = nn.MSELoss()
        if criterion == "L1":
            self.criterion = nn.L1Loss()

        self.optimizer = torch.optim.Adam(
            self.parameters, lr=optim_learning_rate, weight_decay=optim_weight_decay
        )

        self.scheduler = None
        if decay:
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)

        self.n_iterations = n_iterations

        self.model_name = model_name

    def show_graph_info(self, graph) -> int:
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())

        n_features = len(list(nx.get_node_attributes(graph, "features").values())[0])

        print(
            f"The graph has {n_nodes} nodes and {n_edges} edges. Each node has {n_features} features."
        )
        return n_features

    def train(
        self, epochs=1000, train_graph=None, train_label_tensors=None, VERBOSITY=1
    ):
        if train_graph is None:
            train_graph = self.train_graph

        if train_label_tensors is None:
            train_label_tensors = self.train_label_tensors

        previous_loss = 0
        t_0 = time()

        # Calculate the hidden states
        initial_hidden_states = {}
        for node in train_graph.nodes():
            volume = train_graph.nodes[node]["features"][0]
            temperature = train_graph.nodes[node]["features"][1]

            # You can choose your own initialization method here
            initial_hidden_states[node] = torch.tensor([temperature, volume]).to(
                self.device
            )

        if VERBOSITY >= 1:
            self.csv_path = f"logs/LOSSES_{self.model_name}_0.csv"
            c = 1
            while os.path.exists(self.csv_path):
                self.csv_path = f"logs/LOSSES_{self.model_name}_{c}.csv"
                c += 1
            print(f"Writing logs to: {self.csv_path}")
            with open(self.csv_path, "w") as f:
                f.write("Epoch,Total_Loss,Diff,Total_Error,Avg_Error\n")

        # Start training epochs
        for epoch in range(epochs):
            try:
                start_time = time()

                # Message passing
                hidden_states = initial_hidden_states.copy()
                for i in range(self.n_iterations):
                    hidden_states = message_passing_iteration(
                        train_graph,
                        hidden_states,
                        self.message_function,
                        aggregation_function,
                        self.update_function,
                    )

                # Readout module
                regression_outputs = self.readout_module(
                    train_graph, hidden_states, self.readout_function
                )

                losses = {
                    node: self.criterion(
                        regression_outputs[node], train_label_tensors[node]
                    )
                    for node in train_graph.nodes()
                }

                errors = {
                    node: abs(
                        regression_outputs[node] - train_label_tensors[node]
                    ).item()
                    for node in train_graph.nodes()
                }

                self.optimizer.zero_grad()
                total_loss = sum(losses.values())

                total_error = sum(errors.values())
                average_error = total_error / len(errors.values())
                total_loss.backward()
                self.optimizer.step()
                # This is for backwards compatibility with older model versions
                try:
                    if self.scheduler is not None and (epoch + 1 % 100 == 0):
                        self.scheduler.step()
                        self.scheduler.print_lr()
                except AttributeError:
                    self.scheduler = None

                time_per_epoch = (time() - t_0) / (epoch + 1)
                remaining_epochs = epochs - (epoch + 1)
                eta = time_per_epoch * remaining_epochs
                hours = eta // 3600
                minutes = (eta % 3600) // 60
                seconds = eta % 60

                if VERBOSITY >= 1:
                    print(
                        f"{self.model_name}\n"
                        f"\tEpoch {epoch + 1}/{epochs}, Total Loss: {total_loss.item()}, Diff: {previous_loss - total_loss}\n"
                        f"\tTotal Error: {total_error:.1f}, Average Error: {average_error:.2f}\n"
                        f"\tThis epoch took {time()-start_time:.1f} seconds.\n"
                        f"\tETA: {int(hours)}:{int(minutes)}:{int(seconds)} (hh:mm:ss)"
                    )

                if VERBOSITY >= 1:
                    with open(self.csv_path, "a") as f:
                        write_epoch = epoch + 1
                        write_total_loss = total_loss.item()
                        write_diff = previous_loss - total_loss
                        f.write(
                            f"{write_epoch},{write_total_loss},{write_diff},{total_error:.1f},{average_error:.2f}\n"
                        )

                if VERBOSITY >= 2:
                    if (epoch % 100 == 0) or (epoch == epochs - 1):
                        with open(f"logs/{self.model_name}_{epoch}.log", "w") as f:
                            f.writelines(
                                [
                                    f"{node}: {regression_outputs[node].item():.2f} ({train_label_tensors[node].item():.2f})\n"
                                    for node in train_graph.nodes()
                                ]
                            )

                previous_loss = total_loss

                if average_error < 0.1:
                    break

            except KeyboardInterrupt:
                if input("Do you want to save the model?").lower() in ["y", "yes", "1"]:
                    self.save(
                        f"models/interrupt_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_E_{epoch+1}.pickle"
                    )
                return

    def evaluate(self, test_graph=None, test_label_tensors=None):
        if test_graph is None:
            test_graph = self.test_graph

        if test_label_tensors is None:
            test_label_tensors = self.test_label_tensors

        initial_hidden_states = {}
        for node in test_graph.nodes():
            volume = test_graph.nodes[node]["features"][0]
            temperature = test_graph.nodes[node]["features"][1]

            # You can choose your own initialization method here
            initial_hidden_states[node] = torch.tensor([temperature, volume]).to(
                self.device
            )

        hidden_states = initial_hidden_states.copy()

        # Message passing
        for i in range(self.n_iterations):
            hidden_states = message_passing_iteration(
                test_graph,
                hidden_states,
                self.message_function,
                aggregation_function,
                self.update_function,
            )

        # Readout module
        regression_outputs = self.readout_module(
            test_graph, hidden_states, self.readout_function
        )

        for node in test_graph.nodes():
            print(
                regression_outputs[node].item(),
                test_label_tensors[node].item(),
                abs(regression_outputs[node].item() - test_label_tensors[node].item()),
            )

        losses = {
            node: self.criterion(regression_outputs[node], test_label_tensors[node])
            for node in test_graph.nodes()
        }
        errors = {
            node: abs(regression_outputs[node] - test_label_tensors[node])
            for node in test_graph.nodes()
        }

        total_loss = sum(losses.values())
        total_error = sum(errors.values())
        average_error = total_error / len(errors.values())

        return (total_loss.item(), total_error.item(), average_error.item())

    def predict(self, graph: nx.Graph):
        pass

    def save(self, path):
        print(f"Saving this to {path}...")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print("Saved!")

    def load(path):
        with open(path, "rb") as f:
            loaded = CPU_Unpickler(f).load()
            print(type(loaded))
            # loaded.device = (
            #     "cuda"
            #     if torch.cuda.is_available()
            #     else "mps"
            #     if torch.backends.mps.is_available()
            #     else "cpu"
            # )
            loaded.device ="cpu"
            return loaded


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
