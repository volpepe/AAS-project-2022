# Common imports
import os
import time
import math
from typing import List
from curiosity import ICM
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# Vizdoom
import vizdoom as vzd
# Our modules
from variables import PRETRAINING_MAP_PATH, CONFIG_EXTENSION, PRETRAINING_EPISODES,\
    SKIP_FRAMES, TIMESTEPS_PER_EPISODE
from agent import Agent
from state import StateManager
from action import Action


# Obtain the list of available actions as one-hot-encoded buttons
def make_actions(num_available_actions):
    actions = []
    for i in range(num_available_actions):
        ll = [False]*num_available_actions
        ll[i] = True
        actions.append(ll)
    return actions


def play_game(game:vzd.DoomGame, agent:Agent, actions:List, curiosity_model:ICM):
    # Start counting the playing time
    time_start = time.time()
    # Loop over episodes
    for ep in range(PRETRAINING_EPISODES):
        print(f"----------------------\nEpoch: {ep}\n----------------------")
        # Creates the StateManager handling preprocessing of images and the screen buffer
        # for stacking frames into a single state
        state_manager = StateManager()
        # Start new episode of the game
        game.new_episode()
        done = game.is_episode_finished()
        # Run the game until it's over
        timestep = 0
        with tqdm(total=math.ceil(TIMESTEPS_PER_EPISODE/SKIP_FRAMES)) as pbar:
            while not done:
                # Obtain the state from the game (the image on screen)
                screen = game.get_state().screen_buffer
                # Update the StateManager to obtain the current state
                state = state_manager.get_current_state(screen)
                # Let the agent choose the action from the State
                action = agent.choose_action(state)
                # Increase timestep for current episode
                timestep += 1
                pbar.update(1)
                # Apply the action on the game and get the extrinsic reward.
                extrinsic_reward = game.make_action(actions[action], SKIP_FRAMES)
                # Check if we have reached the ending state
                done = game.is_episode_finished()
                if done:
                    # Final state reached: create a simple black image as next image 
                    # (must be uint8 for PIL to work)
                    next_screen = np.zeros(screen.shape, dtype=np.uint8)
                else:
                    # Get next image from the game
                    next_screen = game.get_state().screen_buffer
                # Create the next state
                next_state = state_manager.get_current_state(next_screen)
                # Get the intrinsic reward by the ICM. At the same time, batch the representations
                intrinsic_reward = curiosity_model((tf.cast(tf.expand_dims(state.repr, axis=0), dtype=tf.float32), 
                                                    tf.expand_dims(tf.one_hot(action, depth=len(Action), dtype=tf.float32), axis=0), 
                                                    tf.cast(tf.expand_dims(next_state.repr, axis=0), dtype=tf.float32)))
                print(intrinsic_reward)
                print(extrinsic_reward)
                print("==============")
                reward = extrinsic_reward + intrinsic_reward
                # Train the agent with the current experience batch
                agent.train_step((state, action, reward, next_state))

        total_reward = game.get_total_reward()
        print(f"Got a total reward of: {total_reward}")


if __name__ == '__main__':
    # Check if a GPU is available for training
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("A GPU is available")
        device = "/GPU:0"
    else:
        print("No GPU available")
        device = "/CPU:0"

    # Create a DoomGame instance. 
    # DoomGame represents an instance of the game and provides an interface for our agents to interact with it.
    game = vzd.DoomGame()
    # Load the pre-training map using the configuration files
    training_config_file = PRETRAINING_MAP_PATH + CONFIG_EXTENSION
    game.load_config(training_config_file)
    # Initialize the game.
    game.init()

    # Create the actions as one-hot encoded lists of pressed buttons:
    actions = make_actions(len(Action))

    # Start playing
    with tf.device(device):
        # Everything is executed in the context of the device (on GPU if available or on CPU).
        # Initialize the agent (initially just a random agent)
        agent = Agent()
        # Instantiate the ICM (Curiosity model)
        curiosity_model = ICM(len(Action))
        if os.path.exists(os.path.join('models', 'ICM')):
            # TODO: Load weights
            pass
        # Play the game
        play_game(game, agent, actions, curiosity_model)
        # At the end, close the game
        game.close()
