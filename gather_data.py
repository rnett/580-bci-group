import argparse
import asyncio
import json
import random
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import h5py
import numpy as np

from lib.cortex import Cortex

parser = argparse.ArgumentParser()

parser.add_argument("output_file",
                    type=str,
                    help="File to save the data to.  Should be a .hdf5 file")

parser.add_argument("--frames", type=int, required=False,
                    help="Number of frames of data to save (otherwise goes until killed)")

parser.add_argument("--test", action='store_true', help="Run in testing mode (without the BCI, saves zeroes)")

parser.add_argument("--cortex_creds", type=str,
                    default='./cortex_creds',
                    help="Location of cortex creds (default is ./cortex_creds)")


class Command(Enum):
    Nothing = 0
    Forward = 1
    Backward = 2
    Right = 3
    Left = 4


async def _init(cortex):
    """
    Initializes EMOTIV BCI, creates a session, and starts a subscription for bandpower data.
    Make sure device is connected via EMOTIV App using bluetooth dongle.
    Parameters
    ----------
    cortex : Cortex object
        initialized with developer's credits
    Returns
    -------
    cortex : Cortex object
        updated object with session information
    """

    print("** USER LOGIN **", flush=True)
    await cortex.get_user_login()
    print("** GET CORTEX INFO **", flush=True)
    await cortex.get_cortex_info()
    print("** HAS ACCESS RIGHT **", flush=True)
    await cortex.has_access_right()
    print("** REQUEST ACCESS **", flush=True)
    await cortex.request_access()
    print("** AUTHORIZE **", flush=True)
    await cortex.authorize()
    print("** GET LICENSE INFO **", flush=True)
    await cortex.get_license_info()

    # print("** CONNECT DEVICE **", flush=True)
    # await cortex.connect_device(sensor_id)
    print("** QUERY HEADSETS **", flush=True)
    connected = False
    while (not connected):
        resp = await cortex.query_headsets()
        print(resp['result'][0]['status'])
        connected = (resp['result'][0]['status'] == 'connected')

    print("** CREATE SESSION **", flush=True)
    if len(cortex.headsets) > 0:
        await cortex.create_session(activate=True, headset_id=cortex.headsets[0])

    print("** SUBSCRIBE POW **", flush=True)
    await cortex.subscribe(['pow'])
    return cortex


async def get_data(cortex):
    """
    Gets data using modified version of lib.cortex's get_data().
    Writes the data to given file in csv format.
    Parameters
    ----------
    cortex : Cortex object
        initialized and has a session opened and subscription started
    Returns
    -------
    none
    """
    data_json = await cortex.get_data()
    data_pow = json.loads(data_json)['pow']
    return np.asarray(data_pow)


def display_command(command: Command):
    print(command.name)
    pass  # TODO


def display_frames(frames: int):
    pass  # TODO

#clear canvas and print direction
def draw(w: Canvas, direction: str):
    w.delete("all")
    w.create_text(w.width / 2,
              w.height / 2,
              font=("Purisa", 42),
              text=str)
              
# Command handler
def commandHandler(w: Canvas, command):
    if command == 1:
        draw(w, "FORWARD")
    elif command == 2:
        draw(w, "BACKWARD")
    elif command == 3:
        draw(w, "RIGHT")
    elif command == 4:
        draw(w, "LEFT")

async def main():
    args = parser.parse_args()

    out_file = Path(args.output_file)
    out_file = out_file.with_suffix(".hdf5")

    frames = args.frames

    features = []
    labels = []

    # Set window
    canvas_width = 800
    canvas_height = 600

    master = Tk()

    # Create window        
    w = Canvas(master,
            width=canvas_width, 
            height=canvas_height)

    w.pack()

    # Initialize window text
    w.create_text(canvas_width / 2,
                canvas_height / 2,
                font=("Purisa", 42),
                text="Which direction?")

    if args.test:
        cortex = None
    else:
        cortex = Cortex('./cortex_creds')
        cortex = await _init(cortex)

    try:
        i = 0

        commands = list(Command)
        current_command = Command.Nothing
        end_of_current = datetime.now() + timedelta(seconds=10)

        while frames is None or i < frames:
            # change to a new command
            if datetime.now() > end_of_current:
                if current_command is not Command.Nothing:
                    new_current = Command.Nothing
                    end_of_current = datetime.now() + timedelta(seconds=random.randint(2, 4))
                else:
                    new_current = random.choice(commands)
                    end_of_current = datetime.now() + timedelta(seconds=random.randint(5, 10))

                if new_current != current_command:
                    current_command = new_current
                    display_command(current_command)


            if args.test:
                features.append(np.zeros((14,)))
            else:
                features.append(await get_data(cortex))

            # Note that I'm including Nothing as label[0]
            label = np.zeros((len(list(Command)),), 'float32')
            label[current_command.value] = 1
            labels.append(label)

            # display_frames(i)
            commandHandler(w, i)

            i += 1

            if args.test:
                await asyncio.sleep(0.5)
    finally:
        with h5py.File(str(out_file)) as f:
            f.create_dataset("features", data=np.stack(features, axis=0))
            f.create_dataset("labels", data=np.stack(labels, axis=0))


if __name__ == '__main__':
    asyncio.run(main())
