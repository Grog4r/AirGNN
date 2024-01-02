import json
import math
import os
import pickle

import networkx as nx
from appartement.appartement import Appartement
from appartement.room import Door, Outside, Room


class AppartementParser:
    def __init__(self, base_dir: str):
        self.app = Appartement()
        self.base_dir = base_dir

    def process(self) -> Appartement:
        num_configs = len(os.listdir(self.base_dir))
        grid_size = math.ceil(math.sqrt(num_configs))
        SPACING = 5

        for i, config_file in enumerate(os.listdir(self.base_dir)):
            temp_app = Appartement()
            prefix = config_file.split(".")[0]
            config_file_path = os.path.join(self.base_dir, config_file)
            if os.path.isdir(config_file_path):
                continue
            print(config_file_path)
            with open(config_file_path, "r") as cfg:
                config = json.load(cfg)

            outside_temperature = config["outside_temperature"]
            outside_area = 100_000_000.0
            outsinde_height = 100_000.0

            outside_room = Outside(
                room_name_with_prefix(prefix, "outside"),
                outside_area,
                starting_temperature=outside_temperature,
                room_height=outsinde_height,
            )
            self.app.add_room(outside_room)
            temp_app.add_room(outside_room)

            for room_config in config["rooms"]:
                room = Room(
                    name=room_name_with_prefix(prefix, room_config["name"]),
                    room_area=room_config["room_area"],
                    starting_temperature=room_config["starting_temperature"],
                    window=room_config["has_window"],
                )

                self.app.add_room(room)
                temp_app.add_room(room)
                if room_config["has_window"]:
                    self.app.add_door(
                        Door(outside_room, room, is_open=room_config["window_open"])
                    )
                    temp_app.add_door(
                        Door(outside_room, room, is_open=room_config["window_open"])
                    )

            for door_config in config["doors"]:
                room_from = [
                    room
                    for room in self.app.rooms
                    if room.name == room_name_with_prefix(prefix, door_config["room_1"])
                ][0]
                room_to = [
                    room
                    for room in self.app.rooms
                    if room.name == room_name_with_prefix(prefix, door_config["room_2"])
                ][0]
                door = Door(
                    room_1=room_from, room_2=room_to, is_open=door_config["is_open"]
                )
                self.app.add_door(door)
                temp_app.add_door(door)

            x_pos = i % grid_size
            y_pos = i // grid_size
            center = (x_pos * SPACING, y_pos * SPACING)
            print(f"Center for {config_file}: {center}")
            pos = temp_app.compute_pos(center)
            self.app.add_pos(pos)


def room_name_with_prefix(prefix: str, room_name: str) -> str:
    return f"{prefix}_{room_name}"


if __name__ == "__main__":
    SHOW_GRAPHS = False

    for subdir in ["train", "test"]:
        CONFIG_DIR = f"appartements/json_configs/{subdir}/"
        if not os.path.exists(CONFIG_DIR):
            raise FileNotFoundError(
                "You first have to create the config jsons. See the README for more information."
            )
        OUTPUT_DIR = f"appartements/graphs/{subdir}/"
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        app_parser = AppartementParser(CONFIG_DIR)
        app_parser.process()
        # app_parser.app.show()
        app_parser.app.compute_room_features()

        print(nx.get_node_attributes(app_parser.app.G, name="features"))

        # Run the simulation
        time_counter = 0
        while app_parser.app.update_rooms():
            time_counter += 1

        labels = app_parser.app.compute_room_labels()
        with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
            json.dump(labels, f)

        if SHOW_GRAPHS:
            app_parser.app.show()

        with open(os.path.join(OUTPUT_DIR, "nx_Graph.pickle"), "wb") as f:
            pickle.dump(app_parser.app.G_MINIMAL, f)

        print(f"Done with all rooms after: {time_counter} steps.")
        if SHOW_GRAPHS:
            app_parser.app.show()
