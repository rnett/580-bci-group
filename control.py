from enum import Enum

import cozmo
import time
from cozmo.robot import Robot
from cozmo.util import degrees, distance_mm, speed_mmps


class Controls(Enum):
    Forward = 0
    Right = 1
    Left = 2

def main_loop(robot: Robot):
    # robot.say_text("Hello World").wait_for_completed()
    # Drive forwards for 150 millimeters at 50 millimeters-per-second.
    # robot.drive_straight(distance_mm(1500), speed_mmps(5000)).wait_for_completed()

    robot.drive_wheels(50, -50)

    time.sleep(3)

    # Turn 90 degrees to the left.
    # Note: To turn to the right, just use a negative number.
    # robot.turn_in_place(degrees(90)).wait_for_completed()


if __name__ == '__main__':
    cozmo.run_program(main_loop)