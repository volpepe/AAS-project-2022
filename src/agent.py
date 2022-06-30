from action import Action
from state import State
import random

class Agent():
    '''
    Implementation of a base agent that chooses a random action and does not learn.
    '''
    def __init__(self) -> None:
        pass

    def choose_action(self, state:State) -> Action:
        return random.randint(0, len(Action)-1)

    def train_step(self, train_sequence) -> None:
        pass