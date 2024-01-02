from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from .room import Door, Room, calculate_and_apply_heat_exchange


class Appartement:
    def __init__(self) -> None:
        self.rooms: List[Room] = []
        self.doors: List[Door] = []
        self.G = nx.Graph()
        self.G_MINIMAL = nx.Graph()
        self.positions = {}

    def add_room(self, room: Room) -> None:
        self.rooms.append(room)
        self.G.add_node(room.name)
        self.G_MINIMAL.add_node(room.name)

    def add_rooms(self, rooms: List[Room]) -> None:
        for room in rooms:
            self.add_room(room)

    def add_door(self, door: Door) -> None:
        self.doors.append(door)
        edge = (door.room_1.name, door.room_2.name)
        self.G.add_edge(*edge, color=door.color, style=door.style)
        if door.is_open:
            self.G_MINIMAL.add_edge(*edge, color=door.color, style=door.style)

    def add_doors(self, doors: List[Door]) -> None:
        for door in doors:
            self.add_door(door)

    def update_rooms(self, verbose=False) -> bool:
        """Updates all the rooms temperatures.
        :return: True if there were room temperature changes since the last update.
        """
        changed = False
        for door in self.doors:
            if door.is_open:
                if verbose:
                    print(f"DOOR: {door.room_1.name}-{door.room_2.name}")
                exchange_was_big_enough = calculate_and_apply_heat_exchange(
                    door.room_1, door.room_2
                )
                if verbose:
                    print(exchange_was_big_enough)
                changed = changed or exchange_was_big_enough
        return changed

    def compute_room_features(self):
        for node_id in self.G.nodes():
            # print(f"Generating features for {node_id}...")
            room = [room for room in self.rooms if room.name == node_id][0]
            features = room.get_room_features()

            node_data = {}
            node_data[node_id] = {"features": room.get_room_features_list()}

            nx.set_node_attributes(self.G, node_data)
            nx.set_node_attributes(self.G_MINIMAL, node_data)

    def compute_room_labels(self) -> dict:
        labels = {}
        for node_id, node_data in self.G.nodes(data=True):
            # print(f"Generating label for {node_id}...")
            room = [room for room in self.rooms if room.name == node_id][0]
            label = room.temperature
            node_data["label_temperature"] = label
            labels[node_id] = label
        return labels

    def add_pos(self, pos_dict: dict) -> None:
        self.positions.update(pos_dict)

    def compute_pos(self, center: Tuple[int, int]) -> dict:
        return nx.circular_layout(self.G, center=center)

    def show(self, show_minimal=False, positions=None, with_labels=True):
        graph = self.G_MINIMAL if show_minimal else self.G

        label_dict = {}
        for room in self.rooms:
            label_dict[room.name] = str(room)

        edge_colors = [graph.get_edge_data(*edge)["color"] for edge in graph.edges]
        edge_styles = [graph.get_edge_data(*edge)["style"] for edge in graph.edges]

        plt.title(f"{'Minimal ' if show_minimal else 'Entire '}Graph")

        if positions is None:
            positions = self.positions

        nx.draw(
            graph,
            labels=label_dict,
            pos=positions,
            with_labels=with_labels,
            edge_color=edge_colors,
            style=edge_styles,
            node_size=20,
        )
        plt.show()

    def to_stellar_graph(self):
        sg_graph = sg.StellarGraph.from_networkx(
            self.G_MINIMAL,
            node_type_default="room",
            edge_type_default="door",
            node_features="features",
        )

        print(sg_graph.info())
        return sg_graph
