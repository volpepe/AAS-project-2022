from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras import layers
from action import Action
from state import State

from agent import Agent
from variables import GAMMA, SIGMA

class BaselineActorCriticAgent(Agent):
    """
    Actor-critic network, taken from the curiosity paper.

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
        self.conv1 = layers.Conv2D(filters=8, kernel_size=3, strides=(2,2), padding='same', activation='relu')  # Original: 32 filters, elu activation
        self.conv2 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu') 
        self.conv3 = layers.Conv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu')   
        self.conv4 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.permutation = layers.Permute((3, 1, 2), input_shape=(1, 3, 3, 32))                               # Original connects the CNNs with an LSTM somehow
        self.reshape = layers.Reshape((32, 9))
        self.lstm  = layers.LSTM(64)
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

    def choose_action(self, state:State, training=False) -> Dict:
        action_probs, state_value = self(tf.expand_dims(tf.cast(state.repr, tf.float32), axis=0), training=training)
        # Sample from the actions probability distribution
        action = tf.random.categorical(action_probs, 1)
        return {
            'action': Action(action.numpy()[0]),
            'policy': action_probs[0],
            'value' : state_value[0]
        }

    def compute_loss(self, st:State, a:Dict, st1:State, r:tf.Tensor, a1:Dict, done:bool, iteration:int, tape:tf.GradientTape):
        """
        In actor critic with baseline, the update is computed in this way:
        - We call delta the difference between the immediate reward (the sum of intrinsic and extrinsic)
          summed with the prediction of the value of the next state AND the prediction of the value of the
          current state. 
          
          `delta = R + discount*V_pred(St+1) - V_pred(St)`      (if `St+1 `is terminal, then `V_pred(St+1)=0`)
        
        - The loss function for the actor is:
          
          `delta*-log(action_probs)`

        - For the critic we use a DQN-like approach:
          
          `huber_loss(R + discount*V_pred(St+1), V_pred(St))`   (or `MSE`)

        - To these losses, we add an entropy loss that helps the agent to produce smoother policies
          
          `-sum(action_probs*log(action_probs))`

          We want to keep a high entropy, because it means that no value will dominate over the other unless
          the model is very sure about its choice. It will keep the probability distribution "uncertain" letting
          the agent try different things.
        """
        # Gather all variables we need
        v_st_pred = a['value']
        a_probs = a['policy']
        a_log_probs = tf.math.log(a_probs)
        a_mask = tf.one_hot(a['action'].value, depth=self.num_actions)     # Index of the action the agent chose
        # Check if next state is final
        if not done:
            with tape.stop_recording():
                # Do not record this operation in the gradient tape, we don't want to compute the gradient
                # of this second function call.
                v_st1_pred = self.choose_action(st, training=False)['value']
        else:
            v_st1_pred = tf.zeros((1,))
        # Compute delta
        target = r + self.discount*v_st1_pred
        delta = target - v_st_pred
        # Actor loss
        actor_loss = tf.reduce_sum(self.discount**iteration*delta*-a_log_probs*a_mask)
        # Critic loss
        critic_loss = tf.keras.losses.huber(target, v_st_pred, delta=1.0)
        # Entropy loss
        entropy_loss = -tf.reduce_sum(a_log_probs*a_probs)
        # Total loss
        total_loss = tf.reduce_sum(actor_loss + .5 * critic_loss + SIGMA*entropy_loss)
        return total_loss
