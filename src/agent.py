import tensorflow as tf
from tensorflow import keras
from typing import Tuple

from state import State, StateManager

class Agent(keras.Model):
    '''
    Implementation of a base agent that chooses a random action and does not learn.
    '''
    def __init__(self, num_actions:int, optimizer:tf.keras.optimizers.Optimizer,
            model_name:str='random') -> None:
        super().__init__()
        self.num_actions = num_actions
        self.optimizer = optimizer  # Never used
        self.model_name = model_name

    def call(self, inputs) -> Tuple[tf.Tensor]:
        # Create random logits and their probability distribution
        act_logits = tf.expand_dims(tf.random.uniform((self.num_actions,), minval=-3, maxval=3), axis=0)
        act_probs  = tf.nn.softmax(act_logits)
        return act_logits, act_probs

    def play_one_step(self, env, state:State, episode_step:int, state_manager:StateManager) -> \
           Tuple[State, float, bool]:
        # Add batch dimension
        input_tensor = tf.expand_dims(state.repr, axis=0)
        # Run the model to obtain the (random) action logits and probabilities
        act_logits, act_probs = self(input_tensor)
        # Sample action index from the logits
        action = tf.random.categorical(act_logits, 1)[0,0].numpy()  # Unpack
        # Execute action
        next_obs, reward, done, _ = env.step(action)
        # Get the next state
        next_state = state_manager.get_current_state(next_obs['rgb'])
        return next_state, reward, done

    def training_step(self, episode:int, episode_step:int, global_step:int):
        # This network does not learn!!
        pass
