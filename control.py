import asyncio
import random
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

import cozmo
import time

from cozmo.objects import LightCube
from cozmo.robot import Robot
from cozmo.util import degrees, distance_mm, speed_mmps
from tensorflow.keras.models import load_model
from tensorflow_core.python.keras import Model

from commands import Command
from gather_data import _init, get_data
from lib.cortex import Cortex
from model import inference_model
import numpy as np


def main_loop(robot: Robot):
    global model
    global cortex
    global f_mean
    global f_std

    while True:
        frame = get_data(cortex)

        frame = (frame - f_mean) / f_std
        frame = np.log(frame)

        inferred = model.predict_on_batch(frame[np.newaxis, :])[0]
        command = list(Command)[int(np.argmax(inferred))]
        print(command)
        # command = random.choice(list(Command))

        if not robot.has_in_progress_actions:
            if command is Command.Nothing:
                pass
            elif command is Command.Forward:
                robot.drive_straight(distance_mm(100), speed_mmps(200))
            elif command is Command.Backward:
                robot.drive_straight(distance_mm(-100), speed_mmps(200))
            elif command is Command.Left:
                robot.turn_in_place(degrees(90))
            elif command is Command.Right:
                robot.turn_in_place(degrees(-90))
        else:
            # robot is already doing command, just wait and record signal
            pass


parser = ArgumentParser()
parser.add_argument("model_file", type=str, help="Model to use for inference")

if __name__ == '__main__':

    args = parser.parse_args()
    model_file = Path(args.model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    f_mean = np.load(model_file / "mean.npy")
    f_std = np.load(model_file / "std.npy")

    train_model = load_model(str(model_file))
    model = inference_model(train_model)

    cortex = Cortex('./cortex_creds')
    loop = asyncio.new_event_loop()
    cortex = loop.run_until_complete(_init(cortex))
    asyncio.set_event_loop(loop)

    cozmo.run_program(main_loop)
