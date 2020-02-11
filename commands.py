from enum import Enum


class Command(Enum):
    Nothing = 0
    Forward = 1
    Backward = 2
    Right = 3
    Left = 4