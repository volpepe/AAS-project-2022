from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers
from agent import Agent
from variables import GAMMA

class BaselineActorCritic(Agent):
    """
    Actor-critic based network, implemented using the specifications of the curiosity paper.

    - Input: a batch of 42x42x4 tensors representing the previous 4 42x42 stacked black-and-white frames from the game
    - Output: a tuple containing:
        - action probabilities
        - state value 
    """
    def __init__(self, num_actions:int, name:str='actor_critic') -> None:
        super().__init__(num_actions)
        self.num_actions = num_actions
        self.name = name
        # Instantiate convolutional layers
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu') 
        self.conv3 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')   
        self.conv4 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        # Adapt from 3x3x32 (final convolutional map size computed using 42x42 as an input size for the image) to 32x9
        self.permutation = layers.Permute((3, 1, 2), input_shape=(3, 3, 32))
        self.reshape = layers.Reshape((32, 9))
        self.lstm  = layers.LSTM(128)
        self.flatten = layers.Flatten()
        self.actor = layers.Dense(self.num_actions)        # Produce logits
        self.critic = layers.Dense(1)                      # Produce the state-value directly

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        #  3x3x32 <- 6x6x16 <- 11x11x16 <- 21x21x8 <- 42x42x4
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        # Transformation to connect the convolutional network to the LSTM
        #     32x9   <-   32x3x3
        x = self.reshape(self.permutation(x))
        # We use an LSTM to process the sequence
        x = self.lstm(x)                                # 128
        # Then we produce the policy values
        action_logits = self.actor(x)                   # num_actions
        action_probs  = tf.nn.softmax(action_logits)    # num_actions probabilities
        # Avoid producing a tensor containing probability 0 for some actions.
        action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
        # ... and the state value.
        state_value = self.critic(x)                    # 1
        return action_logits, action_probs, state_value
