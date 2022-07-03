from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers
from action import Action
from state import State

from agent import Agent

class BaselineActorCriticAgent(Agent):
    """
    Actor-critic network, taken from the curiosity paper.

    - Input: a 1x42x42x4 tensor representing the previous 4 42x42 stacked black-and-white frames from the game
    - Output: a tuple containing:
        - action probabilities
        - state value 
    """
    def __init__(self, num_actions) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.conv1 = layers.Conv2D(filters=8, kernel_size=3, strides=(2,2), padding='same', activation='relu')  # Original: 32 filters, elu activation
        self.conv2 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu') 
        self.conv3 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu')   
        self.conv4 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.permutation = layers.Permute((3, 1, 2), input_shape=(1, 3, 3, 32))                               # Original connects the CNNs with an LSTM somehow
        self.reshape = layers.Reshape((32,9))
        self.lstm  = layers.LSTM(64)
        #self.flatten = layers.Flatten()
        self.actor = layers.Dense(self.num_actions, activation='softmax')        # Produce probabilities
        self.critic = layers.Dense(1)                                            # Produce the state-value directly


    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        #  1x3x3x32 <- 1x6x6x16 <- 1x11x11x16 <- 1x21x21x8 <- 1x42x42x4
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        x = self.dropout(x)
        # Transformation to connect the convolutional network to the LSTM
        #     1x32x9   <-   1x32x3x3
        x = self.reshape(self.permutation(x))
        # We use an LSTM to process the sequence
        x = self.lstm(x)                                # 1x64
        # Then we produce the policy values
        action_probs = self.actor(x)                    # 1xnum_actions
        # ... and the state value.
        state_value = self.critic(x)                    # 1x1
        return action_probs, state_value


    def get_state_value_and_action_probs(self, state:State) -> Tuple[tf.Tensor, tf.Tensor]:
        action_probs, state_value = self(tf.expand_dims(tf.cast(state.repr, tf.float32), axis=0))
        return action_probs, state_value


    def choose_action(self, state:State, training=False) -> Action:
        action_probs, state_value = self.get_state_value_and_action_probs(state)
        if training:
            # After having chosen the action, the tensors become available calling "model.last_state_value"
            # and "model.last_action_probs"
            self.last_state_value  = state_value
            self.last_action_probs = action_probs
        return Action(tf.argmax(action_probs, axis=1).numpy()[0])


    def compute_loss(self, st:State, a:Action, st1:State, r:float, a1:Action, 
                     done:bool, tape:tf.GradientTape, discount:float=0.99, iteration:int=1):
        # In actor critic with baseline, the update is computed in this way:
        # - We call delta the difference between the immediate reward (the sum of intrinsic and extrinsic)
        #   summed with the prediction of the value of the next state AND the prediction of the value of the
        #   current state. 
        #   delta = R + discount*V_pred(St+1) - V_pred(St)      (if St+1 is terminal, then V_pred(St+1)=0)
        # - The loss function for the actor is:
        #   delta*-log(action_probs)
        # - For the critic we use a DQN-like approach:
        #   huber_loss(R + discount*V_pred(St+1), V_pred(St))   (or MSE)
        
        # Gather all variables we need
        v_st_pred, a_log_probs = self.last_state_value[0], tf.math.log(self.last_action_probs)[:, a.value]        # Only get the probability of the action the agent chose
        # Check if next state is final
        if not done:
            # Temporarily stop recording gradient for computing the value of the next state
            # because we don't need to track it.
            with tape.stop_recording():
                v_st1_pred = self.get_state_value_and_action_probs(st1)[1][0]
        else:
            v_st1_pred = tf.zeros((1, 1))
        # Compute delta
        target = r + discount*v_st1_pred
        delta = target - v_st_pred
        # Actor loss
        actor_loss = discount**iteration*delta*-a_log_probs
        # Critic loss
        critic_loss = tf.keras.losses.huber(target, v_st_pred, delta=1.0)
        # Total loss
        total_loss = tf.reduce_sum(actor_loss + critic_loss)
        return total_loss
