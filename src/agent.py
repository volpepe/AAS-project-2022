from tensorflow import keras
from action import Action
from state import State
import random

class Agent(keras.Model):
    '''
    Implementation of a base agent that chooses a random action and does not learn.
    '''
    def __init__(self) -> None:
        super().__init__()

    def choose_action(self, state:State) -> Action:
        return random.randint(0, len(Action)-1)

    def train_step(self, train_sequence) -> None:
        pass