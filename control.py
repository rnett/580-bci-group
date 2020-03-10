import asyncio
from argparse import ArgumentParser
from pathlib import Path

import cozmo
import numpy as np
from cozmo.robot import Robot
from cozmo.util import degrees, distance_mm, speed_mmps
from tensorflow.keras.models import load_model

from commands import Command
from gather_data import _init, get_data
from lib.cortex import Cortex
from model import inference_model


def main_loop(robot: Robot):
    global model
    global cortex
    global f_mean
    global f_std

    while True:
        frame = get_data(cortex)

        frame = np.log(frame)
        frame = (frame - f_mean) / f_std

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

def new_main_loop():
    global model
    global cortex
    global f_mean
    global f_std

    frames = []
    inputdat = [None]

    while True:
        frame = get_data(cortex)
        frames.append(frame)
        #frame = np.log(frame)
        #frame = (frame - f_mean) / f_std
        if(len(frames) > 30):
            inputdat[0] = np.array(frames[:30])
            #print(inputdat)
            inferred = model.predict(np.array(inputdat))
            #print(inferred)
            command = list(Command)[int(np.argmax(inferred))]
            if(np.amax(inferred) > 0.9):
                print("{:06.4f}: {}".format(np.amax(inferred), command), flush=True)
            else:
                print(command.Nothing, flush=True)
            frames = frames[1:]
            # command = random.choice(list(Command))
            '''
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
            '''

parser = ArgumentParser()
parser.add_argument("model_file", type=str, help="Model to use for inference")

if __name__ == '__main__':

    args = parser.parse_args()
    model_file = Path(args.model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    #f_mean = np.load(model_file / "mean.npy")
    #f_std = np.load(model_file / "std.npy")

    model = load_model(str(model_file))
    #model = inference_model(train_model)

    cortex = Cortex('./cortex_creds')
    loop = asyncio.new_event_loop()
    cortex = loop.run_until_complete(_init(cortex))
    asyncio.set_event_loop(loop)

    new_main_loop()
    #cozmo.run_program(new_main_loop)
