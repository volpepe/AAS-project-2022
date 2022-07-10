# Vizdoom + Gym
import imp
from tabnanny import check
import gym
from vizdoom import gym_wrapper
# Common imports
import os
from typing import Dict, List, Sequence, Tuple, Union
from collections import deque
import time
import math
import argparse
import numpy as np
import tensorflow as tf
from tqdm import trange
# Our modules
from agent import Agent
from DQN import DQN
from actor_critic import BaselineActorCritic
from state import StateManager
# Variables
from variables import *

##############       ARGS MANAGEMENT         ##############
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-a', '--algorithm', default='random', 
        choices=['random','reinforce', 'actor_critic','dqn'], type=str,
        help='The type of algorithm to use (random, reinforce, actor_critic, dqn). Default is random.')
    args.add_argument('-s', '--save_weights', action='store_true', 
        help='If active, the new weights will be saved.')
    args.add_argument('-l', '--load_weights', action='store_true',
        help='If active, old weights (if found) are loaded into the models.')
    args.add_argument('-st', '--start_episode', type=int, default=0,
        help='The starting episode for the task. Training will resume from this episode onwards. Default is 0.')
    return args.parse_args()

#############           UTILITIES            ###############
def select_agent(args, actions:List) -> Tuple[Agent, str]:
    '''
    Returns the chosen agent and its save weight path
    '''
    agent = args.agent
    if agent == 'random':
        return Agent(len(actions)), ''
    if agent == 'reinforce' or agent == 'actor_critic':
        # The two algorithms share some similarities, so they are implemented with the same agent
        return BaselineActorCritic(len(actions), name=agent), \
            REINFORCE_WEIGHTS_PATH if agent == 'reinforce' else ACTOR_CRITIC_WEIGHTS_PATH
    if agent == 'dqn':
        return DQN(len(actions)), DQN_WEIGHTS_PATH
    # If we arrive here we have chosen something not implemented
    raise NotImplementedError

def make_actions(num_available_actions:int) -> List:
    '''
    Obtain the list of available actions as one-hot-encoded buttons
    '''
    actions = []
    for i in range(num_available_actions):
        ll = [False]*num_available_actions
        ll[i] = True
        actions.append(ll)
    return actions

def save_weights_and_logs(agent:Agent, extrinsic_rewards:Sequence, save_path: str):
    agent.save_weights(save_path)
    print(f"Agent weights saved successfully at {save_path}")
    np.save(os.path.join(LOGS_DIR, agent.name), np.array(extrinsic_rewards), allow_pickle=True)
    print(f"Rewards history saved successfully at {os.path.join(LOGS_DIR, agent.name)}")

def load_weights(agent:Agent, save_path:str):
    # Select actor weights based on the type of the actor
    if agent.name != 'random':
        try:
            agent.load_weights(save_path)
            print("Loaded weights agent")
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

#################### PLAYING #####################

### MAKE IT SEPARATE FOR MONTE CARLO (REINFORCE) VS ACTOR CRITIC AND DQN (TD)
def play_game(env, agent:Agent, actions:List, save_weights:bool=True, save_path:str='', 
            start_episode:int=0):
    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NO)
    # Keep a list of obtained rewards
    episode_rewards = []
    # Iterate over episodes
    for episode in range(TRAINING_EPISODES):
        # Initialise a list for the episode rewards
        episode_rewards = []
        # Instantiate the state manager for the episode
        state_manager = StateManager()
        # Obtain the first observation by resetting the environment
        initial_obs = env.reset()
        # The initial state is processed by the state manager
        state = state_manager.get_current_state(initial_obs['rgb'])
        done = False
        # Iterate until the episode is over
        with trange(MAX_TIMESTEPS_PER_EPISODE) as pbar:
            for episode_step in pbar:
                # Update the progress bar
                pbar.update(1)
                # Play one step of the game, obtaining the following state, the reward and 
                # whether the episode is finished
                epsilon = max(1- ep_num/500, 0.01)
                state, reward, done, info = play_one_step(env, initial_state, epsilon, state_manager)
                rewards.append(reward)
                if done:
                    break
                elif ep_num > 1:
                    training_step(batch_size)
        episode_reward = sum(ep_rewards)
        print(f"Episode reward: {episode_reward}")


##################### START #####################
if __name__ == '__main__':
    # Collect args
    args = parse_args()

    # Create folders for models and logs
    os.makedirs(os.path.join('models', 'simpler', 'logs'), exist_ok=True)
    os.makedirs(os.path.join('models', 'dqn'), exist_ok=True)
    os.makedirs(os.path.join('models', 'reinforce'), exist_ok=True)
    os.makedirs(os.path.join('models', 'actor_critic'), exist_ok=True)

    # Check if a GPU is available for training
    device = check_gpu()

    # Initialize the environment
    env = gym.make("VizdoomHealthGatheringSupreme-v0", frame_skip=4)
    # There are 3 available actions (move forward, turn left, turn right)
    num_actions = 3
    # The observation space contains 240x320 RGB frames and the health of the player (that we don't need)
    # Create the actions as one-hot encoded lists of pressed buttons:
    actions = make_actions(num_actions)

    # Start playing
    with tf.device(device):
        # Everything is executed in the context of the device (on GPU if available or on CPU).
        # Initialize the agent.
        agent, save_path = select_agent(args, actions)
        # Check if we need to load the weights for the agent and for the ICM
        if args.load_weights:
            load_weights(agent, save_path)
        # Play the game training the agents or evaluating the loaded weights.
        play_game(env, agent, actions, args.save_weights, save_path, args.start_episode)

    # At the end, close the environment
    env.close()