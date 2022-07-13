import os
import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple
from DQN import DQNAgent
from actor_critic import BaselineA2C
from agent import Agent
from variables import ACTOR_CRITIC_WEIGHTS_PATH, CLIP_NO, DQN_WEIGHTS_PATH, LOGS_DIR, LR_A2C, LR_DQN

### Utility functions ###

def select_agent(args, num_actions:int) -> Tuple[Agent, str]:
    '''
    Returns the chosen agent and its save weight path
    '''
    agent = args.algorithm
    if agent == 'random':
        return Agent(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR_DQN, clipnorm=CLIP_NO)), ''
    if agent == 'a2c':
        return BaselineA2C(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR_A2C, clipnorm=CLIP_NO),
            model_name=agent), ACTOR_CRITIC_WEIGHTS_PATH
    if agent == 'dqn':
        return DQNAgent(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR_DQN, clipnorm=CLIP_NO)), \
        DQN_WEIGHTS_PATH
    # If we arrive here we have chosen something not implemented
    raise NotImplementedError

def save_weights_and_logs(agent:Agent, extrinsic_rewards:Sequence, save_path: str):
    if save_path:
        agent.model.save_weights(save_path)
        print(f"Agent weights saved successfully at {save_path}")
    else:
        print(f'Tried to save model weights, but model needs no saving.')
    np.save(os.path.join(LOGS_DIR, agent.model_name), np.array(extrinsic_rewards), allow_pickle=True)
    print(f"Rewards history saved successfully at {os.path.join(LOGS_DIR, agent.model_name)}")

def load_weights(agent:Agent, save_path:str):
    # Select actor weights based on the type of the actor
    if agent.model_name != 'random':
        try:
            agent.model.load_weights(save_path)
            print("Loaded weights of agent")
        except:
            print(f"Could not find weights for the agent at {save_path}")
    else:
        print("Agent not implemented or does not require weights.")

def check_gpu() -> str:
    # Check if a GPU is available for training
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"A GPU is available: {tf.config.list_physical_devices('GPU')}")
        device = "/GPU:0"
    else:
        print("No GPU available: using CPU.")
        device = "/CPU:0"
    return device