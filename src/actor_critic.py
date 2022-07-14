from collections import deque
import math
from typing import Tuple
import numpy as np
import tensorflow as tf
from agent import Agent
from state import State, StateManager
from tensorflow.keras import layers, Model
from variables import CRITIC_C, GAMMA, MAX_TIMESTEPS_PER_EPISODE, SIGMA, SKIP_FRAMES

class ActorCriticModel(Model):
    def __init__(self, num_actions) -> None:
        super(ActorCriticModel, self).__init__()
        # Instantiate convolutional layers
        self.num_actions = num_actions
        self.conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=(4,4), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=(2,2), padding='same', activation='relu') 
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu')   
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation='relu')
        self.actor = layers.Dense(self.num_actions, activation='softmax')   # Produce logits
        self.critic = layers.Dense(1)                                       # Produce the state-value directly

    def call(self, inputs):
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        x = self.conv3(self.conv2(self.conv1(inputs)))
        # We flatten the output and pass it through a dense layer
        x = self.flatten(x)
        x = self.dense(x)
        # Then we produce the policy values
        action_probs = self.actor(x)                    # 1xnum_actions probabilities
        # Avoid producing a tensor containing a probability of exactly 0 or 1 for some actions.
        action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0-(1e-5))
        # ... Then we produce the state value.
        state_value = self.critic(x)                    # 1x1
        return action_probs, state_value

class BaselineA2C(Agent):
    """
    A2C agent, implemented using modified specifications from both the DQN and A3C papers.

    - Input: a batch of 42x42x4 tensors representing the previous 4 42x42 stacked black-and-white frames from the game
    - Output: a tuple containing:
        - action probabilities
        - state value 
    """
    def __init__(self, num_actions:int, optimizer:tf.keras.optimizers.Optimizer,
            model_name:str='a2c') -> None:
        super(BaselineA2C, self).__init__(num_actions, optimizer)
        self.num_actions = num_actions
        self.model_name = model_name
        # Store the optimizer for the model
        self.optimizer = optimizer
        # Store the model
        self.model = ActorCriticModel(self.num_actions)
        # We keep an episode buffer that is used to keep the past knowledge
        self.episode_buffer = deque(maxlen=MAX_TIMESTEPS_PER_EPISODE)

    def play_one_step(self, env, state:State, epsilon:float, state_manager:StateManager) -> \
            Tuple[State, float, bool]:
        # The play step for actor critic is very similar to the one of the random agent:
        # Add batch to state representation
        input_tensor = tf.cast(tf.expand_dims(state.repr, axis=0), tf.float32)
        # Obtain the action probabilities and state values
        policy, value = self.model(input_tensor)
        # Extract the value and the policy
        value = np.squeeze(value.numpy())
        policy = np.squeeze(policy.numpy())
        # The action is chosen by sampling from the policy
        action = np.random.choice(self.num_actions, p=policy)
        # Execute the action, get the next observation and reward
        next_obs, reward, done, _ = env.step(action)
        # Transform the observation into the next state
        next_state = state_manager.get_current_state(next_obs['rgb'])
        # Add the computed values to a buffer for later use in training.
        self.episode_buffer.append((state, action, np.clip(reward, -1, 1), next_state, done))
        return next_state, reward, done

    def collect_experiences(self) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Utility function used to unpack the internal experience buffer into valid arrays of experiences.
        '''
        # Collect the elements we need into 5 different arrays:
        states, actions, rewards, next_states, dones = (
            np.array([experience[i] for experience in self.episode_buffer]) 
            for i in range(5)
        )
        return states, actions, rewards, next_states, dones

    def a2c_update_(self, episode:int, episode_step:int, global_step:int):
        # Retrieve the episode's experience from the buffer
        states, actions, rewards, next_states, dones = self.collect_experiences()
        # First of all, compute the value of the next state and store it as R
        _, R = self.model(tf.cast(tf.expand_dims(next_states[-1].repr, axis=0), tf.float32))
        # The return is 0 if the episode is over
        R = tf.squeeze(R)*(1-dones[-1])
        # We compute the returns going backwards
        returns = np.zeros_like(rewards)
        for t in range(len(rewards)-1, -1, -1):
            R = rewards[t] + GAMMA * R
            returns[t] = R
        # Stack all state representations
        states = np.stack([s.repr for s in states])
        # Open the GradientTape to record the next operations for gradient computation
        with tf.GradientTape() as tape:
            # Pass all states into the actor critic network, obtain action probabiltiies and state values
            action_probs, state_values = self.model(
                tf.cast(states, dtype=tf.float32)
            )
            state_values = tf.squeeze(state_values)
            action_probs = tf.squeeze(action_probs)
            # Advantages are the difference between the computed returns and the values
            advantages = returns - state_values
            # Compute log probabilities
            log_probs = tf.math.log(action_probs)
            # Compute the probability distribution's entropy
            entropy_term = -tf.reduce_sum(action_probs * log_probs)
            # Compute action masks
            action_masks = tf.stack(tf.one_hot(actions, depth=self.num_actions))
            # The actor loss is:
            actor_loss = tf.reduce_mean(advantages * -tf.reduce_sum(log_probs * action_masks, axis=1))
            # Notice that it's a negative loss, because we want to actually improve the weights
            # in the direction of the loss. This is the performance measure J(theta).
            # The critic loss is:
            critic_loss = tf.keras.losses.mean_squared_error(returns, state_values)
            # We also add an entropy loss to stabilize the probability distributions
            # and avoid large spikes. The more we maximise the entropy, the less probable each
            # event becomes, making the action probabilities more distributed.
            total_loss = actor_loss + CRITIC_C * critic_loss - SIGMA * entropy_term
            # We already consider the state value in the critic loss
            tf.stop_gradient(advantages)
        # Compute the gradients and apply changes
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def training_step(self, episode:int, episode_step:int, global_step:int):
        '''
        Update the weights of the network.
        The loss of the actor critic agent is divided in two parts: the actor loss and the critic loss.

        1) The critic is updated in a similar manner to DQN. The loss is the MSE between:
        
        - `Rt + discount * V_pred(St+1)`
        - `V_pred(St)`

        2) We compute delta, the difference between the value obtained summing the reward with the 
            discounted value of the following state (coming from the critic) and the value of the 
            current state (again coming from the critic). 
            The term V_pred(St) is the baseline, which reduces the variance of the updates by reducing 
            the amount of correction.
        
        - `delta = [Rt + discount * V_pred(St+1)] - V_pred(St)`
        
        3) Then we compute the loss for the actor:
        
        `-(delta*log(action_probs))`
        
        4) We add an entropy loss to force the probability distribution of the actions to 
        be flatter and more distributed. The entropy term is computed over all timesteps and summed
        to the total loss (with a small coefficient)
        '''
        # The training step is only executed if we are at the terminal state (last `done` is true or t=T)
        if episode_step == (math.ceil(MAX_TIMESTEPS_PER_EPISODE/SKIP_FRAMES) - 1) or \
                self.episode_buffer[-1][-1]:
            self.a2c_update_(episode, episode_step, global_step)
            # Clear the episode buffer
            self.episode_buffer.clear()
