from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers
from agent import Agent

class DQN(Agent):
    """
    DQN based network, implemented using the specifications of the curiosity paper.

    - Input: a batch of 42x42x4 tensors representing the previous 4 42x42 stacked black-and-white frames from the game
    - Output: the Q value for each state-action pair related to the input state
    """
    def __init__(self, num_actions) -> None:
        super().__init__(num_actions)
        self.num_actions = num_actions
        self.name = 'dqn'
        # Instantiate convolutional layers
        self.conv1 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='elu'),
        self.conv2 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='elu'),
        self.conv3 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='elu'),
        self.conv4 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu'),
        self.permute = layers.Permute((3, 1, 2), input_shape=(3, 3, 32)),
        self.reshape= layers.Reshape((32, 9)),
        self.lstm = layers.LSTM(64),
        self.dense1 = layers.Dense(64, activation="elu"),
        self.dense2 = layers.Dense(32, activation="elu"),
        self.q_values = layers.Dense(self.num_actions)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        #  3x3x32 <- 6x6x16 <- 11x11x16 <- 21x21x8 <- 42x42x4
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        # Transformation to connect the convolutional network to the LSTM
        #     32x9   <-   32x3x3
        x = self.reshape(self.permutation(x))
        # We use an LSTM to process the sequence
        x = self.lstm(x)                                # 128
        # We pass the tensor through some dense layers
        x = self.dense2(self.dense1(x))
        # Finally, we produce the Q values
        q_vals = self.q_values(x)
        return q_vals
