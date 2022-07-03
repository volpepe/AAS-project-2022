import tensorflow as tf
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

    def choose_action(self, state:State, training=False) -> Action:
        return Action(random.randint(0, len(Action)-1))

    def compute_loss(self, st:State, a:Action, st1:State, r:float, a1:Action, 
                     done:bool, tape:tf.GradientTape, discount:float=0.99, iteration:int=1):
        pass