from collections import deque
from typing import Tuple
import tensorflow as tf
from agent import Agent
from state import State, StateManager
from tensorflow.keras import layers
from variables import GAMMA, MAX_AC_EXP_BUFFER_LEN, MAX_TIMESTEPS_PER_EPISODE, SIGMA

class BaselineActorCritic(Agent):
    """
    Actor-critic based network, implemented using the specifications of the curiosity paper.

    - Input: a batch of 42x42x4 tensors representing the previous 4 42x42 stacked black-and-white frames from the game
    - Output: a tuple containing:
        - action probabilities
        - state value 
    """
    def __init__(self, num_actions:int, optimizer:tf.keras.optimizers.Optimizer,
            model_name:str='actor_critic') -> None:
        super(BaselineActorCritic, self).__init__(num_actions, optimizer)
        self.num_actions = num_actions
        self.model_name = model_name
        # Store the loss functions and the optimizer for the model
        self.critic_loss_fn = tf.keras.losses.MeanSquaredError()
        self.actor_loss_fn = lambda x: x    # Fake function because we compute a loss value directly
        self.optimizer = optimizer
        # We keep an episode buffer that is used to keep all the knowledge of an episode in case the agent is a 
        # REINFORCE agent, or to keep the last transition in case the agent is an Actor Critic.
        self.episode_buffer = deque(maxlen=1 if self.model_name == 'actor_critic' else MAX_AC_EXP_BUFFER_LEN)
        # Instantiate convolutional layers
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu') 
        self.conv3 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')   
        self.conv4 = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256)
        self.actor = layers.Dense(self.num_actions)        # Produce logits
        self.critic = layers.Dense(1)                      # Produce the state-value directly

    def call(self, inputs: tf.Tensor):
        # Input is a 1x42x42x4 sequence of images. We first apply some convolutions (1 is the batch size)
        #  3x3x32 <- 6x6x16 <- 11x11x16 <- 21x21x8 <- 42x42x4
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        x = self.flatten(x)                             # 128
        x = self.dense(x)                               # 256
        # Then we produce the policy values
        action_logits = self.actor(x)                   # num_actions
        action_probs  = tf.nn.softmax(action_logits)    # num_actions probabilities
        # Avoid producing a tensor containing probability 0 for some actions.
        action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
        # ... and the state value.
        state_value = self.critic(x)                    # 1
        return action_logits, action_probs, state_value

    def play_one_step(self, env, state:State, episode_step:int, state_manager:StateManager) -> \
            Tuple[State, float, bool]:
        # The play step for actor critic is very similar to the one of the random agent:
        # Add batch to state representation
        input_tensor = tf.expand_dims(state.repr, axis=0)
        # Obtain the action logits, probabilities and state values
        action_logits, _, _ = self(input_tensor)
        # The action is chosen by sampling the action logits
        action = tf.random.categorical(action_logits, 1).numpy()[0,0]   # Unpack
        # Execute the action, get the next observation and reward
        next_obs, reward, done, _ = env.step(action)
        # Transform the observation into the next state
        next_state = state_manager.get_current_state(next_obs['rgb'])
        # Add this transition to a buffer for later use in training.
        self.episode_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done

    def actor_critic_update_(self, episode:int):
        # Retrieve last experience from the buffer
        state, action, reward, next_state, done = self.episode_buffer[0]
        # Predict the following state value (don't track this computation)
        _, _, next_val = self.predict(tf.expand_dims(
            tf.cast(next_state.repr, tf.float32), axis=0), 
            verbose=0
        )
        # Compute target: Rt if St+1 is terminal or Rt + discount * next_Q_val_max if it's not
        target = reward + (GAMMA * next_val * ~done)
        # Open the GradientTape to record the next operations
        with tf.GradientTape() as tape:
            # Compute the current state value and action probabilities (policy) for the state
            _, action_probs, state_value = self(
                tf.cast(tf.expand_dims(state.repr, axis=0), dtype=tf.float32)
            )
            # The critic loss is:
            critic_loss = tf.reduce_mean(self.critic_loss_fn(target, state_value))
            # Compute delta
            delta = target - state_value
            # Compute log probabilities
            log_pi_ats = tf.math.log(action_probs)
            # Get the probability of the actual action that the agent has executed
            log_pi_at = tf.gather(log_pi_ats, tf.expand_dims(action, axis=0), batch_dims=1)
            # The actor loss is:
            actor_loss = tf.reduce_mean(-self.actor_loss_fn(delta*log_pi_at))
            # Notice that it's a negative loss, because we want to actually improve the weights
            # in the direction of the loss. This is the performance measure J(theta).
            # We also add an entropy loss to stabilize the probability distributions
            # and avoid large spikes. The more we maximise the entropy, the less probable each
            # event becomes, making the action probabilities more distributed.
            entropy_loss = SIGMA*-tf.reduce_mean(log_pi_ats*action_probs)
            # Total loss is:
            loss = actor_loss + critic_loss + entropy_loss
        # Compute the gradients and apply changes
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    def training_step(self, episode:int):
        '''
        Update the weights of the network.
        For now, we only implemented the actor critic update step. #TODO.

        - Actor Critic

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
            be flatter and more distributed.
            
            `entropy_loss = -reduce_sum(a_log_probs*a_probs)`
        '''
        if self.model_name == 'actor_critic':
            self.actor_critic_update_(episode)
        else:
            raise NotImplementedError()
        # Clear episode buffer
        self.episode_buffer.clear()
