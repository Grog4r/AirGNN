import argparse
import json
import os

from model import GNN_Model, load_graph


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        help="The path to the model to retrain.",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1000,
        help="Number of retraining epochs (default: 1000)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"The path to the model {args.model} was not found.")

    model_path = args.model
    model_name = model_path.split("/")[-1]
    model = GNN_Model.load(model_path)

    epochs = args.epochs

    model.train(epochs=epochs)
    model.save(f"models/RETRAINED_{model_name}")
