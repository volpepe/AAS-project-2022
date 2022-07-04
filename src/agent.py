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
    def __init__(self, num_actions) -> None:
        super().__init__()
        self.num_actions = num_actions

    def choose_action(self, state:State, training=False) -> Dict:
        act = Action(random.randint(0, len(Action)-1))
        policy = tf.one_hot(act.value, depth=self.num_actions),
        return {
            'action': Action(random.randint(0, len(Action)-1)),
            'policy': policy
        }

    def compute_loss(self, st:State, a:Dict, st1:State, r:tf.Tensor, a1:Dict, 
                     done:bool, iteration:int=1):
        return 0