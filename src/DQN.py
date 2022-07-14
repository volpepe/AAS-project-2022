from collections import deque
from typing import Tuple
import numpy as np
import tensorflow as tf
from state import State, StateManager
from tensorflow.keras import layers, Model, backend
from agent import Agent
from variables import DQN_BATCH_SIZE, DQN_C, DQN_START_UPDATES_EPISODE, \
    DQN_TRAIN_EVERY, GAMMA, INPUT_SHAPE, MAX_DQN_EXP_BUFFER_LEN

class DQNModel(Model):
    """
    DQN based network, implemented using the specifications of the DQN paper
    (Human-level control through deep reinforcement learning by Mnih et al., https://www.nature.com/articles/nature14236.pdf ). 

    - Input: a batch of 84x84x4 tensors representing the previous 4 84x84 stacked black-and-white frames from the game
    - Output: the Q value for each state-action pair related to the input state
    """
    def __init__(self, num_actions) -> None:
        super(DQNModel, self).__init__()
        self.num_actions = num_actions
        # Instantiate convolutional layers
        self.conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=(4,4), activation='relu')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=(2,2), activation='relu')
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation="relu")
        self.q_values = layers.Dense(num_actions)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        x = self.conv3(self.conv2(self.conv1(inputs)))
        # Flatten the output
        x = self.flatten(x)
        # We pass the tensor through a dense layer
        x = self.dense(x)
        # Finally, we produce the Q values
        q_vals = self.q_values(x)
        return q_vals

def ClippedMSE(y_true, y_pred):
    '''
    MSE loss function taken directly from Keras, but modified to add the 
    clipping of the difference as in the paper.
    '''
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return backend.mean(tf.math.square(tf.clip_by_value(y_pred - y_true, -1., 1.)), axis=-1)

class DQNAgent(Agent):

    def __init__(self, num_actions:int, optimizer:tf.keras.optimizers.Optimizer,
            model_name:str='dqn') -> None:
        super().__init__(num_actions, optimizer)
        self.num_actions = num_actions
        self.model_name = model_name
        # The DQN model has a replay buffer that uses to collect experience for training
        self.replay_buffer = deque(maxlen=MAX_DQN_EXP_BUFFER_LEN)
        # Store the loss function and the optimizer for the model
        self.loss_function = ClippedMSE
        self.optimizer = optimizer
        # We instantiate the target network and the actual Q-value network
        self.target_Q_model = DQNModel(num_actions)
        self.model = DQNModel(num_actions)
        
    def update_target_network(self):
        # Copy the same weights
        self.target_Q_model.set_weights(self.model.get_weights())

    def epsilon_greedy_policy(self, state:State, epsilon:float) -> int:
        '''
        Return a random action, or the action obtained by a greedy policy on the Q values.
        '''
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            # Obtain the Q values for each action given the current state
            Q_values = self.model(tf.cast(tf.expand_dims(state.repr, axis=0), dtype=tf.float32))
            # Act greedily with respect to them
            return np.argmax(Q_values[0])   # Remove batch dimension, get index with highest Q value

    def play_one_step(self, env, state:State, epsilon:float, state_manager:StateManager) -> \
            Tuple[State, float, bool]:
        # Compute the action according to an epsilon greedy policy
        action = self.epsilon_greedy_policy(state, epsilon)
        # Execute the action, get the next observation and reward
        next_obs, reward, done, _ = env.step(action)
        # Transform the observation into the next state
        next_state = state_manager.get_current_state(next_obs['rgb'])
        # Add all the information into the replay buffer in order to train on it later.
        # Note that saved rewards are clipped in the range -1, 1
        self.replay_buffer.append((state, action, np.clip(reward, -1., 1.), next_state, done))
        return next_state, reward, done

    def sample_experiences(self) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Utility function used to sample the internal replay buffer and fill a batch of experience.
        '''
        # Sample `batch_size` random indices from the replay buffer
        indices = np.random.randint(0, len(self.replay_buffer), size=DQN_BATCH_SIZE)
        batch = [self.replay_buffer[index] for index in indices]
        # Collect the elements we need into 5 different arrays:
        states, actions, rewards, next_states, dones = (
            np.array([experience[i] for experience in batch]) 
            for i in range(5)
        )
        return states, actions, rewards, next_states, dones

    def training_step(self, episode:int, episode_step:int, global_step:int):
        '''
        Update the weights of the network. 
        Given a full transition St -> At -> Rt -> St+1, the loss function
        in DQN is the MSE between two quantities:

        - `Rt + discount * max(Q_values(St+1))`
        - `Q_values(St)[At]`

        In other words, we use the fact that the Q values for state St and action At
        should be as close as possible to the sum of the reward with the following
        Q value. 
        Since DQN does off-policy updates, we take the maximum Q value of the next state
        rather than also computing the next action (At+1).

        Note: the direction of the update (the gradient) is `grad(Q_values(St)[At])`:
        we don't compute the gradient of the approximation of the next Q values.
        '''
        # We only start updating after some episodes to fill the buffer a little first
        if episode >= DQN_START_UPDATES_EPISODE:
            # Also, we update once every DQN_TRAIN_EVERY
            if (global_step % DQN_TRAIN_EVERY) == 0:
                # Sample a batch of experience from the experience buffer
                states, actions, rewards, next_states, dones = self.sample_experiences()
                # Predict the following Q values using the target network
                next_Q_vals = self.target_Q_model.predict(tf.stack(
                    [tf.cast(st_1.repr, tf.float32) for st_1 in next_states]), 
                    batch_size=DQN_BATCH_SIZE, verbose=0)
                # Collect the maximum Q values (keeping the batch dimension)
                next_Q_vals_max = np.max(next_Q_vals, axis=1)
                # Compute target: Rt if St+1 is terminal or Rt + discount * next_Q_val_max if it's not
                targets = rewards + (GAMMA * next_Q_vals_max * (1-dones))
                # Create a one hot encoded mask to gather the Q value for the action the agent actually chose
                actions_mask = tf.one_hot(actions, depth=self.num_actions)
                # Open a GradientTape to record the following operations: we need to compute their gradients later
                with tf.GradientTape() as tape:
                    # Compute the Q values using the model
                    Q_vals = self.model(tf.cast(
                        tf.stack([state.repr for state in states]),
                        dtype=tf.float32))                                              # batch_sizex3
                    # Get the specific Q value for the actions the agent has made
                    Q_vals = tf.squeeze(tf.reduce_sum(Q_vals*actions_mask, axis=1, keepdims=True))
                    # Compute the loss between the targets (of which we do not compute gradients) and
                    # the Q values obtained by the model for state St
                    loss = tf.reduce_sum(self.loss_function(targets, Q_vals))
                # Compute gradients and apply them on the trainable variables. Gradient is computed only with respect
                # to the non-target network.
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            # If C steps have been made, update the target network.
            if (global_step % DQN_C) == 0:
                self.update_target_network()

