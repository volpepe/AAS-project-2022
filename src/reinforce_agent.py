from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras import layers
from action import Action
from state import State
from agent import Agent
from variables import GAMMA, SIGMA

class BaselineREINFORCEAgent(Agent):
    """
    REINFORCE-based network, implemented using the specifications of the curiosity paper.

    - Input: a 1x42x42x4 tensor representing the previous 4 42x42 stacked black-and-white frames from the game
    - Output: a tuple containing:
        - action probabilities
        - state value 
    """
    def __init__(self, num_actions, optimizer, discount:float=GAMMA) -> None:
        super().__init__(num_actions, optimizer)
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.discount = discount
        self.conv1 = layers.Conv2D(filters=8, kernel_size=3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))  # Original: 32 filters, elu activation
        self.conv2 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)) 
        self.conv3 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))   
        self.conv4 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))
        self.dropout = layers.Dropout(0.2)
        self.permutation = layers.Permute((3, 1, 2), input_shape=(3, 3, 32))
        self.reshape = layers.Reshape((32, 9))
        self.lstm  = layers.LSTM(64)
        self.dense = layers.Dense(64)
        self.flatten = layers.Flatten()
        self.actor = layers.Dense(self.num_actions)        # Produce logits
        self.critic = layers.Dense(1)                      # Produce the state-value directly

    def call(self, inputs: tf.Tensor, lstm_active=True) -> Tuple[tf.Tensor, tf.Tensor]:
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        #  1x3x3x32 <- 1x6x6x16 <- 1x11x11x16 <- 1x21x21x8 <- 1x42x42x4
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        x = self.dropout(x)
        if lstm_active:
            # Transformation to connect the convolutional network to the LSTM
            #     1x32x9   <-   1x32x3x3
            x = self.reshape(self.permutation(x))
            # We use an LSTM to process the sequence
            x = self.lstm(x)                                # 1x64
        else:
            x = self.flatten(x)
            x = self.dense(x)
        x = self.dropout(x)
        # Then we produce the policy values
        action_logits = self.actor(x)                    # 1xnum_actions
        action_probs  = tf.nn.softmax(action_logits)     # 1xnum_actions probabilities
        # Avoid producing a tensor containing probability 0 for some actions.
        action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
        # ... and the state value.
        state_value = self.critic(x)                    # 1x1
        return action_logits, action_probs, state_value

    def choose_action(self, state:State, training=False, lstm_active=True) -> Dict:
        action_logits, action_probs, state_value = self(tf.expand_dims(tf.cast(state.repr, tf.float32), axis=0), 
            training=training, lstm_active=lstm_active)
        # Sample from the actions probability distribution
        action = tf.random.categorical(action_logits, 1)
        return {
            'action': Action(action.numpy()[0]),
            'policy': action_probs[0],
            'value' : state_value[0]
        }
