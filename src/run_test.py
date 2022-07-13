# Vizdoom + Gym
import numpy as np
import gym
from vizdoom import gym_wrapper
# Common imports
import math
from time import sleep
import argparse
import tensorflow as tf
from tqdm import trange
# Our modules
from agent import Agent
from state import StateManager
from utils import check_gpu, load_weights, select_agent
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

#############            PLAYING             ###############
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
        with trange(math.ceil(MAX_TIMESTEPS_PER_EPISODE/SKIP_FRAMES)) as pbar:
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
    num_actions = 7
    
    # The observation space contains 240x320 RGB frames

    # The rewards we get are:
    # +dX for getting closer to the vest.
    # -dX for getting further from the vest.
    # -100 death penalty

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