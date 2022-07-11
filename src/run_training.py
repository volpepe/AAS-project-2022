# Vizdoom + Gym
import gym
from vizdoom import gym_wrapper
# Common imports
import math
import os
from typing import List, Sequence, Tuple
import argparse
import numpy as np
import tensorflow as tf
from tqdm import trange
# Our modules
from agent import Agent
from DQN import DQNAgent
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
    args.add_argument('-s', '--save_weights', action='store_true', 
        help='If active, the new weights will be saved.')
    args.add_argument('-l', '--load_weights', action='store_true',
        help='If active, old weights (if found) are loaded into the models.')
    args.add_argument('-st', '--start_episode', type=int, default=0,
        help='The starting episode for the task. Training will resume from this episode onwards. Default is 0.')
    return args.parse_args()

#############           UTILITIES            ###############
def select_agent(args, num_actions:int) -> Tuple[Agent, str]:
    '''
    Returns the chosen agent and its save weight path
    '''
    agent = args.algorithm
    if agent == 'random':
        return Agent(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NO)), ''
    if agent == 'a2c':
        # The two algorithms share some similarities, so they are implemented with the same agent
        return BaselineA2C(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NO),
            model_name=agent), ACTOR_CRITIC_WEIGHTS_PATH
    if agent == 'dqn':
        return DQNAgent(num_actions, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIP_NO)), \
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

#################### PLAYING #####################
### MAKE IT SEPARATE FOR MONTE CARLO (REINFORCE) VS ACTOR CRITIC AND DQN (TD)
def play_game_TD(env, agent:Agent, save_weights:bool=True, save_path:str='', 
            start_episode:int=0):
    # Keep a list of obtained rewards
    game_rewards = []
    # Keep track of the global timestep
    global_timestep = 0
    # Iterate over episodes
    for episode in range(start_episode, TRAINING_EPISODES):
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
                pbar.set_description(f'Global timestep: {global_timestep}')
                # Compute epsilon in case the policy for training is epsilon-greedy. We use epsilon-decay to reduce random
                # actions during time. Initially, epsilon is very high, but it quickly decreases.
                # As decay, we multiply the initial epsilon by EPS_D at each timestep until it reaches the minimum
                epsilon = max(EPS_S*EPS_D**(global_timestep), EPS_MIN)  
                global_timestep += 1
                # Play one step of the game, obtaining the following state, the reward and 
                # whether the episode is finished
                next_state, reward, done = agent.play_one_step(env, state, 
                    epsilon, state_manager)
                state = next_state
                episode_rewards.append(reward)
                # Try to do a training step
                agent.training_step(episode, episode_step, global_timestep)
                # Check if the episode is over and end the episode
                if done:
                    break
        # The episode is over: sum the obtained rewards
        episode_reward = sum(episode_rewards)
        game_rewards.append({global_timestep: episode_reward})
        print(f"Episode: {episode}, Total reward: {episode_reward}")
        recent_mean_reward = np.mean([list(g.values())[0] for g in game_rewards[-10:]])
        print(f"Mean reward in last 10 episodes: {recent_mean_reward:.2f}")
        # Check if it's time to save the weights
        if save_weights and ((episode % WEIGHTS_SAVE_FREQUENCY) == 0 or episode == (TRAINING_EPISODES-1)):
            save_weights_and_logs(agent, game_rewards, save_path)

##################### START #####################
if __name__ == '__main__':
    # Collect args
    args = parse_args()

    # Create folders for models and logs
    os.makedirs(os.path.join('models', 'simpler', 'logs'), exist_ok=True)
    os.makedirs(os.path.join('models', 'simpler', 'dqn'), exist_ok=True)
    os.makedirs(os.path.join('models', 'simpler', 'a2c'), exist_ok=True)

    # Check if a GPU is available for training
    device = check_gpu()

    # Initialize the environment
    env = gym.make("VizdoomCorridor-v0", frame_skip=SKIP_FRAMES)

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
        # If we are using actor critic or DQN as an agent, we need to update in a TD fashion.
        if agent.model_name == 'dqn' or agent.model_name == 'a2c' or agent.model_name == 'random':
            play_game_TD(env, agent, args.save_weights, save_path, args.start_episode)
        # Otherwise, we use a Monte-Carlo update style where we only update the network at the end
        #   of the episode
        else:
            raise NotImplementedError

    # At the end, close the environment
    env.close()