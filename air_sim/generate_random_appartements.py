import json
import os
import random
from typing import Dict, List


def get_room_names(rooms: List[Dict]) -> List[str]:
    return_list = []
    for room in rooms:
        return_list.append(room["name"])
    return return_list


def valid_door(room_1, room_2, doors) -> bool:
    if room_1 == room_2:
        return False

    for door in doors:
        if door["room_1"] == room_1 and door["room_2"] == room_2:
            return False

        if door["room_2"] == room_1 and door["room_1"] == room_2:
            return False

    return True


def all_rooms_connected(rooms, doors):
    for room in rooms:
        room_connected = False
        for door in doors:
            if room["name"] in [door["room_1"], door["room_2"]]:
                room_connected = True
        if not room_connected:
            return False
    return True


if __name__ == "__main__":
    configs_to_create = {
        "appartements/json_configs/train": 100,
        "appartements/json_configs/test": 20,
    }
    for base_dir, NUM_APPARTEMENTS_TO_CREATE in configs_to_create.items():
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for i in range(NUM_APPARTEMENTS_TO_CREATE):
            num = i + 1

            file_name = f"app_{num}.json"

            print(f"Creating appartement in '{file_name}'.")

            json_dict = {}

            outside_temperature = round(random.uniform(-10.0, 35.0), 2)
            json_dict["outside_temperature"] = outside_temperature

            json_dict["rooms"] = []
            n_rooms = 0
            while True:
                if random.random() <= 0.25 and n_rooms >= 5:
                    break
                print("Creating a new room:")
                n_rooms += 1
                name = f"room_{n_rooms+1}"
                room_area = random.uniform(2.0, 25.0)

                starting_temperature = random.uniform(15.0, 25.0)

                has_window = random.random() > 0.5
                if has_window:
                    window_open = random.random() > 0.5
                else:
                    window_open = False

                room_dict = {
                    "name": name,
                    "room_area": room_area,
                    "starting_temperature": starting_temperature,
                    "has_window": has_window,
                    "window_open": window_open,
                }
                print(room_dict)
                json_dict["rooms"].append(room_dict)

            json_dict["doors"] = []
            n_doors = 0
            while True:
                if all_rooms_connected(json_dict["rooms"], json_dict["doors"]):
                    break
                print("Creating a new door:")
                n_doors += 1

                found_good_door = False
                while not found_good_door:
                    room_1 = random.choice(get_room_names(json_dict["rooms"]))
                    room_2 = random.choice(get_room_names(json_dict["rooms"]))
                    found_good_door = valid_door(room_1, room_2, json_dict["doors"])

                is_open = random.random() > 0.5
                door_dict = {
                    "room_1": room_1,
                    "room_2": room_2,
                    "is_open": is_open,
                }
                print(door_dict)
                json_dict["doors"].append(door_dict)

            with open(os.path.join(base_dir, file_name), "w") as f:
                f.write(json.dumps(json_dict, indent=True))
