class Room:
    def __init__(
        self,
        name: str,
        room_area: float,
        starting_temperature: float,
        room_height: float = 3.0,
        window: bool = False,
    ) -> None:
        """Initilaizes a room

        :param name: The name of the room
        :param room_area: The area of the room
        :param starting_temperature: The temperature of the room in the beginning
        :param room_height: The height of the room, defaults to 3.0
        :param window: If the room has a window, defaults to False
        """
        self.name = name
        self.volume = room_area * room_height  # m³
        DENSITY = 1.204  # kg/m³
        self.mass = DENSITY * self.volume  # kg
        self.temperature = starting_temperature  # °C
        self.window = window

    def __str__(self):
        return f"{self.name}\n{self.temperature:.2f}°C"

    @property
    def temperature_kelvin(self) -> float:
        return self.temperature + 273.15

    def update_temperature(self, delta_temp: float) -> float:
        self.temperature += delta_temp
        return self.temperature

    def get_room_features(self) -> dict:
        return {
            "volume": self.volume,
            "temperature": self.temperature,
            "has_window": self.window,
        }

    def get_room_features_list(self) -> list:
        return [self.volume, self.temperature]


class Outside(Room):
    pass


def calculate_and_apply_heat_exchange(
    room_1: Room, room_2: Room, verbose=False
) -> bool:
    if verbose:
        print(f"--> {room_1.name} -> {room_2.name}")
    temp_1_k = room_1.temperature_kelvin
    temp_2_k = room_2.temperature_kelvin
    delta_T1 = temp_2_k - temp_1_k
    delta_T2 = temp_1_k - temp_2_k
    H = 0.5
    A = 2
    temp_1_update = delta_T1 * H * A / room_1.volume
    temp_2_update = delta_T2 * H * A / room_2.volume
    if verbose:
        print(f"T1 update: {temp_1_update:.3f} K, T2 update: {temp_2_update:.3f} K")
    temp_1 = room_1.temperature
    temp_2 = room_2.temperature
    temp_1_new = room_1.update_temperature(temp_1_update)
    temp_2_new = room_2.update_temperature(temp_2_update)
    if verbose:
        print(f"{temp_1:.2f} -> {temp_1_new:.2f}; {temp_2:.2f} -> {temp_2_new:.2f}")
    return (abs(temp_1_update) + abs(temp_2_update)) > 0.002


class Door:
    def __init__(self, room_1: Room, room_2: Room, is_open: bool = True) -> None:
        self.room_1 = room_1
        self.room_2 = room_2
        self.is_open = is_open

    @property
    def color(self) -> str:
        if isinstance(self.room_1, Outside) or isinstance(self.room_2, Outside):
            return "blue"
        else:
            return "black"
            return "red"

    @property
    def style(self) -> str:
        if self.is_open:
            return "solid"
        else:
            return "dotted"

    def open_door(self):
        self.is_open = True

    def close_door(self):
        self.is_open = False

    def invert_door(self):
        self.is_open = not self.is_open
