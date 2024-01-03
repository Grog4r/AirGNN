import argparse
import json
import pickle

from model import GNN_Model, load_graph


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")

    parser.add_argument(
        "-hm",
        "--hidden_message",
        type=int,
        default=1,
        help="Number of hidden units in the message layer (default: 1)",
    )

    parser.add_argument(
        "-hu",
        "--hidden_update",
        type=int,
        default=32,
        help="Number of hidden units in the update layer (default: 32)",
    )

    parser.add_argument(
        "-hr",
        "--hidden_readout",
        type=int,
        default=16,
        help="Number of hidden units in the readout layer (default: 16)",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)",
    )

    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay for the optimizer (default: 0)",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs (default: 500)",
    )

    parser.add_argument(
        "-i",
        "--n_iterations",
        type=int,
        default=7,
        help="Number of message passing iterations (default: 7)",
    )

    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        default="relu",
        help="The activation function to use. Can be either relu or selu. (default: relu)",
    )

    parser.add_argument(
        "-c",
        "--criterion",
        type=str,
        default="MSE",
        help="The loss function to use. Can be either MSE or L1. (default: MSE)",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        help="The verbosity level to use. (default: 1)",
    )

    parser.add_argument(
        "-d",
        "--decay",
        action="store_true",
        help="Enable decaying of learning_rate. (default: False)",
    )

    parser.add_argument(
        "-iw",
        "--init_weights",
        action="store_true",
        help="Enable initializing weights with xavier_uniform. (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train_graph = load_graph("appartements/graphs/train/nx_Graph.pickle")
    test_graph = load_graph("appartements/graphs/test/nx_Graph.pickle")

    train_labels = json.load(open("appartements/graphs/train/labels.json"))
    test_labels = json.load(open("appartements/graphs/test/labels.json"))

    args = parse_arguments()

    HIDDEN_MESSAGE = args.hidden_message
    HIDDEN_UPDATE = args.hidden_update
    HIDDEN_READOUT = args.hidden_readout
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    N_ITERATIONS = args.n_iterations
    ACTIVATION = args.activation
    CRITERION = args.criterion
    DECAY = args.decay
    EPOCHS = args.epochs

    MODEL_NAME = (
        f"M{HIDDEN_MESSAGE}"
        f"-U{HIDDEN_UPDATE}"
        f"-R{HIDDEN_READOUT}"
        f"-LR{LEARNING_RATE}"
        f"-WD{WEIGHT_DECAY}"
        f"-I{N_ITERATIONS}"
        f"-A{ACTIVATION}"
        f"-C{CRITERION}"
        f"-D{DECAY}"
        f"-EP{EPOCHS}"
    )
    print(f"Model Name: {MODEL_NAME}")

    model = GNN_Model(
        train_graph=train_graph,
        train_labels=train_labels,
        test_graph=test_graph,
        test_labels=test_labels,
        hidden_message=HIDDEN_MESSAGE,
        hidden_update=HIDDEN_UPDATE,
        hidden_readout=HIDDEN_READOUT,
        optim_learning_rate=LEARNING_RATE,
        optim_weight_decay=WEIGHT_DECAY,
        n_iterations=N_ITERATIONS,
        activation=ACTIVATION,
        criterion=CRITERION,
        decay=DECAY,
        model_name=MODEL_NAME,
    )

    model.train(epochs=EPOCHS, VERBOSITY=args.verbosity)
    model.save(f"models/{MODEL_NAME}.pickle")
