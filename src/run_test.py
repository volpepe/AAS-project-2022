# Vizdoom + Gym
import numpy as np
import gym
from vizdoom import gym_wrapper
# Common imports
from time import sleep
from typing import Tuple
import argparse
import tensorflow as tf
from tqdm import trange
# Our modules
from agent import Agent
from DQN import DQN
from actor_critic import BaselineA2C
from state import StateManager
# Variables
from variables import *

##############       ARGS MANAGEMENT         ##############
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-a', '--algorithm', default='random', 
        choices=['random','reinforce', 'a2c','dqn'], type=str,
        help='The type of algorithm to use (random, reinforce, a2c, dqn). Default is random.')
    args.add_argument('-l', '--load_weights', action='store_true',
        help='If active, old weights (if found) are loaded into the models.')
    return args.parse_args()

def select_agent(args, num_actions:int) -> Tuple[Agent, str]:
    '''
    Returns the chosen agent and its save weight path
    '''
    agent = args.algorithm
    if agent == 'random':
        return Agent(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NO)), ''
    elif agent == 'reinforce' or agent == 'a2c':
        # The two algorithms share some similarities, so they are implemented with the same agent
        return BaselineA2C(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NO),
            model_name=agent), ACTOR_CRITIC_WEIGHTS_PATH
    elif agent == 'dqn':
        return DQN(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NO)), \
        DQN_WEIGHTS_PATH
    # If we arrive here we have chosen something not implemented
    raise NotImplementedError

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

def play_game(env, agent:Agent):
    # Iterate over episodes
    for episode in range(TESTING_EPISODES):
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
                env.render(mode = "human")
                # Play one step of the game, obtaining the following state, the reward and 
                # whether the episode is finished
                next_state, reward, done = agent.play_one_step(env, state, 
                    epsilon=0, state_manager=state_manager)
                state = next_state
                episode_rewards.append(reward)
                # Check if the episode is over and end the episode
                if done:
                    break
                sleep(1/30) # 30 FPS
        # The episode is over: sum the obtained rewards
        episode_reward = sum(episode_rewards)
        print(f"Episode: {episode}, Total reward: {episode_reward}")

##################### START #####################
if __name__ == '__main__':
    # Collect args
    args = parse_args()

    # Check if a GPU is available for training
    device = check_gpu()

    # Initialize the environment
    env = gym.make("VizdoomCorridor-v0")

    # There are 7 available actions:
    # MOVE_LEFT
    # MOVE_RIGHT
    # ATTACK
    # MOVE_FORWARD
    # MOVE_BACKWARD
    # TURN_LEFT
    # TURN_RIGHT
    
    # The observation space contains 240x320 RGB frames

    # The rewards we get are:
    # +dX for getting closer to the vest.
    # -dX for getting further from the vest.
    # -100 death penalty
    num_actions = 7

    # Start playing
    with tf.device(device):
        # Everything is executed in the context of the device (on GPU if available or on CPU).
        # Initialize the agent.
        agent, save_path = select_agent(args, num_actions)
        # Run random input in the network to create initial weights
        agent.model(np.stack([np.random.random(INPUT_SHAPE)]))
        if agent.model_name == 'dqn':
            # Also call the target network
            agent.target_Q_model(np.stack([np.random.random(INPUT_SHAPE)]))
        # Check if we need to load the weights for the agent and for the ICM
        if args.load_weights:
            load_weights(agent, save_path)
        # For DQN, copy the weights of the model in the target network.
        if agent.model_name == 'dqn':
            agent.update_target_network()
        # Play the game training the agents or evaluating the loaded weights.
        play_game(env, agent)

    # At the end, close the environment
    env.close()