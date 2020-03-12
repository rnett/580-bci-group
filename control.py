import asyncio
from datetime import datetime, timedelta
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

DELAY_TIME = 5

def main_loop(robot: Robot):
    global model
    global cortex
    global f_mean
    global f_std
    global test
    global sequence_length

    sequence = np.zeros((sequence_length, 70), 'float32')

    last_command_time = datetime.now()
    last_command = None

    while True:
        if not test:
            frame = get_data(cortex)
        else:
            frame = np.zeros((70,), 'float32')

        sequence = sequence[1:]
        sequence = np.concatenate([sequence, frame.reshape(1, -1)], axis=0)

        inferred = model.predict_on_batch(sequence[np.newaxis, :])[0]
        command = list(Command)[int(np.argmax(inferred))]

        conf = float(inferred[int(np.argmax(inferred))])

        print(f"{command} ({last_command}) - {conf}")

        if conf <= 0.93:
            command = Command.Nothing
        # command = random.choice(list(Command))

        if last_command is None or last_command != command:
            last_command = command
            command = Command.Nothing
        else:
            last_command = None
            pass

        if not robot.has_in_progress_actions and datetime.now() > last_command_time:
            if command is Command.Nothing:
                pass
            elif command is Command.Forward:
                robot.drive_straight(distance_mm(100), speed_mmps(200))
                last_command_time = datetime.now() + timedelta(seconds=DELAY_TIME)
            elif command is Command.Backward:
                robot.drive_straight(distance_mm(-100), speed_mmps(200))
                last_command_time = datetime.now() + timedelta(seconds=DELAY_TIME)
            elif command is Command.Left:
                robot.turn_in_place(degrees(90))
                last_command_time = datetime.now() + timedelta(seconds=DELAY_TIME)
            elif command is Command.Right:
                robot.turn_in_place(degrees(-90))
                last_command_time = datetime.now() + timedelta(seconds=DELAY_TIME)
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
parser.add_argument("--test", action='store_true', help="Test w/o cortex")

if __name__ == '__main__':

    args = parser.parse_args()
    test = args.test
    model_file = Path(args.model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file {model_file} does not exist")

    model = load_model(str(model_file))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=model.metrics + ["acc"], weighted_metrics=model.metrics + ["acc"])

    sequence_length = model.layers[0].input_shape[1]

    if not test:
        cortex = Cortex('./cortex_creds')

    loop = asyncio.new_event_loop()

    if not test:
        cortex = loop.run_until_complete(_init(cortex))

    asyncio.set_event_loop(loop)

    new_main_loop()
    #cozmo.run_program(new_main_loop)
