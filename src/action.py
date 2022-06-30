from enum import Enum

class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    NO_OP = 5 
    USE = 5         # USE and NO_OP are the same action