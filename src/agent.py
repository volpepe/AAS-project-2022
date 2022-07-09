import tensorflow as tf
from tensorflow import keras
from action import Action
from state import State
import random
from typing import Dict

class Agent(keras.Model):
    '''
    Implementation of a base agent that chooses a random action and does not learn.
    '''
    def __init__(self, num_actions,optimizer) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.optimizer = optimizer

    def choose_action(self, state:State, training=False) -> Dict:
        act = Action(random.randint(0, len(Action)-1))
        policy = tf.one_hot(act.value, depth=self.num_actions),
        return {
            'action': Action(random.randint(0, len(Action)-1)),
            'policy': policy
        }