import argparse
import json
import os

from model import GNN_Model

path = "/home/niklas/Nextcloud/Uni/INFB-10/IoTSeminar/air_sim/AirGnn/data/models/M1-U32-R16-LR0.001-WD0-I6-Arelu-EP500.pickle"



def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        help="The path of the model to evaluate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"The path to the model {args.model} was not found.")

    model_path = args.model
    model_name = model_path.split("/")[-1]
    model = GNN_Model.load(model_path)

    total_loss, total_error, average_error = model.evaluate()

    print(f"The model {model_name} has the follwing metrics:")
    print(f"Loss: {total_loss}, Total Error: {total_error}, Avg. Error: {average_error}")

