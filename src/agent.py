import tensorflow as tf
from tensorflow import keras
from typing import Tuple

class Agent(keras.Model):
    '''
    Implementation of a base agent that chooses a random action and does not learn.
    '''
    def __init__(self, num_actions) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.name = 'random'

    def call(self, inputs) -> Tuple[tf.Tensor]:
        # Create random logits and their probability distribution
        act_logits = tf.random.uniform((self.num_actions,), minval=-3, maxval=3)
        act_probs  = tf.nn.softmax(act_logits)
        return act_logits, act_probs