from enum import IntEnum
import numpy as np
import pickle
from gains import controller_gains


class ConeClasses(IntEnum):
    YELLOW = 0
    BLUE = 1
    ORANGE = 2
    BIG = 3


with open("bin/state_config.bin", "rb") as f:
    state_config = pickle.load(f)

state_config["controller_gains"] = controller_gains

with open("bin/car_params.bin", "rb") as f:
    car_params = pickle.load(f)
