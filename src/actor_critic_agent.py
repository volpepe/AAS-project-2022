import tensorflow as tf
from tensorflow.keras import layers

from typing import Tuple

from action import Action
from agent import Agent
from state import State


class BaselineActorCriticAgent(Agent):
    """
    Actor-critic network, taken from the curiosity paper.

    - Input: a 1x42x42x4 tensor representing the previous 4 42x42 stacked black-and-white frames from the game
    - Output: a tuple containing:
        - action scores
        - state value 
    """
    def __init__(self, num_actions) -> None:
        self.num_actions = num_actions
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu') 
        self.conv3 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')   
        self.conv4 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.permutation = layers.Permute((3, 1, 2), input_shape=(1, 3, 3, 32))
        self.reshape = layers.Reshape((32,9))
        self.lstm  = layers.LSTM(256)
        self.actor = layers.Dense(self.num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        #  1x288 <- 1x3x3x32 <- 1x6x6x32 <- 1x11x11x32 <- 1x21x21x32 <- 1x42x42x4
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        # Transformation to connect the convolutional network to the LSTM
        #     1x32x9   <-   1x32x3x3
        x = self.reshape(self.permutation(x))
        # We use an LSTM to process the sequence
        x = self.lstm(x)                                                                                        # 1x256
        # Then we produce the policy values (not probabilities yet, just scores for the actions)
        action_scores = self.actor(x)                                                                           # 1xnum_actions
        # ... and the state value.
        state_value = self.critic(x)                                                                            # 1x1
        return action_scores, state_value

    def choose_action(self, state: State) -> Action:
        # TODO
        return super().choose_action(state)

    def train_step(self, train_sequence) -> None:
        # TODO
        return super().train_step(train_sequence)   