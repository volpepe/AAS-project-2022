from enum import Enum

class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2
    NO_OP = 3 
    USE = 3         # USE and NO_OP are the same action