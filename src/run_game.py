# Vizdoom
import vizdoom as vzd
# Common imports
from typing import Dict, List, Tuple
from collections import deque
import os
import time
import math
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# Our modules
from agent import Agent
from actor_critic_agent import BaselineActorCriticAgent
from state import StateManager
from action import Action
from curiosity import ICM
# Variables
from variables import ACTOR_CRITIC_WEIGHTS_PATH, ICM_WEIGHTS_PATH, \
    PRETRAINING_MAP_PATH, CONFIG_EXTENSION, PRETRAINING_EPISODES, \
    SKIP_FRAMES, STATS_UPDATE_FREQUENCY, TESTING_EPISODES, TESTING_MAP_PATH_DENSE, \
    TESTING_MAP_PATH_SPARSE, TESTING_MAP_PATH_VERY_SPARSE, TIMESTEPS_PER_EPISODE, TRAINING_EPISODES


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--task', default='train', choices=['train','pretrain','test'], type=str,
        help='What task to start (training, pretraining or test). Default is "train"')
    args.add_argument('-m', '--map', default='dense', choices=['dense','sparse','verysparse'], type=str,
        help='What map to run (dense, sparse or verysparse). Pretrain map is only available using the pretrain argument in --task. Default is "dense"')
    args.add_argument('-a', '--agent', default='random', choices=['random','actor_critic'], type=str,
        help='The type of agent to use (random or actor_critic). Default is random.')
    args.add_argument('-s', '--save_weights', action='store_true', 
        help='If active, the new weights will be saved.')
    args.add_argument('-l', '--load_weights', action='store_true',
        help='If active, old weights (if found) are loaded into the models.')
    args.add_argument('-st', '--start_episode', type=int, default=0,
        help='The starting episode for the task. Training will resume from this episode onwards. Default is 0.')
    args.add_argument('-nr', '--no_render', action='store_true',
        help='If active, the game screen will not appear (useful for eg. server training)')
    return args.parse_args()


# Obtain the list of available actions as one-hot-encoded buttons
def make_actions(num_available_actions):
    actions = []
    for i in range(num_available_actions):
        ll = [False]*num_available_actions
        ll[i] = True
        actions.append(ll)
    return actions


def get_next_state(state_manager, prev_screen):
    # Check if we have reached the ending state
    done = game.is_episode_finished()
    if done:
        # Final state reached: create a simple black image as next image 
        # (must be uint8 for PIL to work)
        next_screen = np.zeros(prev_screen.shape, dtype=np.uint8)
    else:
        # Get next image from the game
        next_screen = game.get_state().screen_buffer
    # Create the next state
    next_state = state_manager.get_current_state(next_screen)
    return done, next_state


def training_step(game:vzd.DoomGame, agent:Agent, actions:List, curiosity_model:ICM, state_manager:StateManager, 
                  optimizer:tf.keras.optimizers.Optimizer, iteration:int=1, discount:float=0.99, policy_update_weight=0.5):
    # Obtain the state from the game (the image on screen)
    screen = game.get_state().screen_buffer
    # Update the StateManager to obtain the current state
    state = state_manager.get_current_state(screen)
    with tf.GradientTape(persistent=True) as tape:          # Persistent because we need to update two separate models
        # Let the agent choose the action from the State
        action = agent.choose_action(state, training=True)
        # Temporarily stop the gradient recording as we compute the next state and extrinsic reward
        with tape.stop_recording():
            # Apply the action on the game and get the extrinsic reward.
            extrinsic_reward = game.make_action(actions[action.value], SKIP_FRAMES)
            done, next_state = get_next_state(state_manager, screen)
        # Resume gradient recording: get the intrinsic reward by the ICM.
        intrinsic_reward = curiosity_model((tf.cast(tf.expand_dims(state.repr, axis=0), dtype=tf.float32), 
                                            tf.expand_dims(tf.one_hot(action.value, depth=len(Action), dtype=tf.float32), axis=0), 
                                            tf.cast(tf.expand_dims(next_state.repr, axis=0), dtype=tf.float32)),
                                            training=True)
        reward = extrinsic_reward + intrinsic_reward
        # Then we need to compute the loss for the agent
        agent.compute_loss(state, action, next_state, reward, None, done, tape, discount, iteration)
        agent_loss = sum(agent.losses)
        icm_loss = sum(curiosity_model.losses)
        # The ICM had already computed its loss in the forward pass.
        total_loss = policy_update_weight*agent_loss + icm_loss

    # Compute gradients and updates
    gradients_curiosity = tape.gradient(total_loss, curiosity_model.trainable_weights)    
    gradients_agent     = tape.gradient(total_loss,           agent.trainable_weights)
    optimizer.apply_gradients(zip(gradients_curiosity, curiosity_model.trainable_weights))
    optimizer.apply_gradients(zip(gradients_agent,               agent.trainable_weights))
    # Remove the tape
    del tape
    return done, {                      # Return if episode is done and some statistics about the step
        'intrinsic_reward': intrinsic_reward, 
        'extrinsic_reward': extrinsic_reward, 
        'total_reward'    : reward,
        'total_loss'      : total_loss,
        'icm_loss'        : icm_loss, 
        'agent_loss'      : agent_loss
    }


def test_step(game:vzd.DoomGame, agent:Agent, actions:List, curiosity_model:ICM, state_manager:StateManager) -> Tuple[bool, Dict]:
    # Obtain the state from the game (the image on screen)
    screen = game.get_state().screen_buffer
    # Update the StateManager to obtain the current state
    state = state_manager.get_current_state(screen)
    # Let the agent choose the action from the State
    action = agent.choose_action(state, training=False)
    # Apply the action on the game and get the extrinsic reward.
    extrinsic_reward = game.make_action(actions[action.value], SKIP_FRAMES)
    done, next_state = get_next_state(state_manager, screen)
    # Get the intrinsic reward by the ICM.
    intrinsic_reward = curiosity_model((tf.cast(tf.expand_dims(state.repr, axis=0), dtype=tf.float32), 
                                        tf.expand_dims(tf.one_hot(action.value, depth=len(Action), dtype=tf.float32), axis=0), 
                                        tf.cast(tf.expand_dims(next_state.repr, axis=0), dtype=tf.float32)),
                                        training=False)
    return done, {
        'intrinsic_reward': intrinsic_reward, 
        'extrinsic_reward': extrinsic_reward, 
        'total_reward'    : intrinsic_reward + extrinsic_reward,
    }


def choose_episodes(task):
    if task == 'train':
        return TRAINING_EPISODES
    elif task == 'pretrain':
        return PRETRAINING_EPISODES
    else:
        return TESTING_EPISODES


def play_game(game:vzd.DoomGame, agent:Agent, actions:List, curiosity_model:ICM, train:bool=True,
              save_weights:bool=False, start_episode:int=0, task:str='train', discount:float=0.99,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)):
    # Create some deques for statistics about the game
    extrinsic_rewards = deque([])
    intrinsic_rewards = deque([])
    total_rewards     = deque([])
    if train:
        icm_losses    = deque([])
        agent_losses  = deque([])
        total_losses  = deque([])
    # Select maximum number of epochs based on task
    tot_episodes = choose_episodes(task)
    # Start counting the playing time
    time_start = time.time()
    # Loop over episodes
    for ep in range(start_episode, tot_episodes):
        print(f"----------------------\nEpoch: {ep}/{tot_episodes-1}\n----------------------")
        # Creates the episode's StateManager handling preprocessing of images and the screen buffer
        # for stacking frames into a single state
        state_manager = StateManager()
        # Start new episode of the game
        game.new_episode()
        done = game.is_episode_finished()
        # Run the game until it's over
        timestep = 1
        with tqdm(total=math.ceil(TIMESTEPS_PER_EPISODE/SKIP_FRAMES)) as pbar:
            while not done:
                if train:
                    done, stats = training_step(game, agent, actions, curiosity_model, state_manager, optimizer, 
                                                timestep, discount, policy_update_weight=0.5)
                else:
                    done, stats = test_step(game, agent, actions, curiosity_model, state_manager)
                # Increase timestep for current episode
                timestep += 1
                pbar.update(1)
            # Is it time to update the statistics array?
            if (timestep-1 % STATS_UPDATE_FREQUENCY) == 0:
                extrinsic_rewards.append(stats['extrinsic_reward'])
                intrinsic_rewards.append(stats['intrinsic_reward'])
                total_rewards.append(stats['total_reward'])
                if train:
                    icm_losses.append(stats['icm_loss'])
                    agent_losses.append(stats['agent_loss'])
                    total_losses.append(stats['total_loss'])
    
        # End of episode: compute aggregated statistics
        icm_stats = curiosity_model.end_episode()
        total_extrinsic_reward = game.get_total_reward()
        print(f"Total reward: {total_extrinsic_reward:.4f}")
        if task == 'train' or task == 'pretrain':
            print(f"Losses:")
            print(f"\tICM Loss: {stats['icm_loss']:.4f}\t (Forward loss: {icm_stats['loss_forward']:.4f}, Inverse loss: {icm_stats['loss_inverse']:.4f})")
            print(f"\tAgent Loss: {stats['agent_loss']}")
            print(f"\tTotal Loss: {stats['total_loss']}")

    # End of episodes
    time_end = time.time()
    print(f"Task finished. It took {time_end-time_start:.4f} seconds.")

    # Save weights if needed
    if save_weights:
        curiosity_model.save_weights(ICM_WEIGHTS_PATH)
        print(f"ICM weights saved succesfully at {ICM_WEIGHTS_PATH}")
        agent.save_weights(ACTOR_CRITIC_WEIGHTS_PATH)
        print(f"Agent weights saved successfully at {ACTOR_CRITIC_WEIGHTS_PATH}")


def select_map(args):
    task = args.task
    game_map = args.map
    # Only one map is available for pretraining
    if task == 'pretrain':
        return PRETRAINING_MAP_PATH + CONFIG_EXTENSION
    # Otherwise choose based on the map
    if game_map == 'dense':
        return TESTING_MAP_PATH_DENSE + CONFIG_EXTENSION
    if game_map == 'sparse':
        return TESTING_MAP_PATH_SPARSE + CONFIG_EXTENSION
    if game_map == 'verysparse':
        return TESTING_MAP_PATH_VERY_SPARSE + CONFIG_EXTENSION
    # If we arrive here we have chosen something not implemented
    raise NotImplementedError


def select_agent(args, actions):
    agent = args.agent
    if agent == 'random':
        return Agent()
    if agent == 'actor_critic':
        return BaselineActorCriticAgent(len(actions))
    # If we arrive here we have chosen something not implemented
    raise NotImplementedError


def load_weights(curiosity_model, agent):
    # Check if path for the ICM weights exists
    if os.path.exists(ICM_WEIGHTS_PATH):
        curiosity_model.load_weights(ICM_WEIGHTS_PATH)
        print("Loaded weights for the ICM")
    else:
        print(f"Could not find weights for the ICM model at {ICM_WEIGHTS_PATH}")
    # Select actor weights based on the type of the actor
    if isinstance(agent, BaselineActorCriticAgent):
        if os.path.exists(ACTOR_CRITIC_WEIGHTS_PATH):
            agent.load_weights(ACTOR_CRITIC_WEIGHTS_PATH)
            print("Loaded weights for the Actor Critic agent")
        else:
            print(f"Could not find weights for the Actor Critic model at {ACTOR_CRITIC_WEIGHTS_PATH}")
    else:
        print("Agent not implemented or does not require weights.")


if __name__ == '__main__':
    # Collect args
    args = parse_args()

    # Check if a GPU is available for training
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"A GPU is available: {tf.config.list_physical_devices('GPU')}")
        device = "/GPU:0"
    else:
        print("No GPU available: using CPU.")
        device = "/CPU:0"

    # Create a DoomGame instance. 
    # DoomGame represents an instance of the game and provides an interface for our agents to interact with it.
    game = vzd.DoomGame()
    # Load the chosen map using the configuration files
    config_file = select_map(args)
    game.load_config(config_file)
    # Overwrite "window_visible" property if needed
    game.set_window_visible(not args.no_render)
    # Initialize the game.
    game.init()

    # Create the actions as one-hot encoded lists of pressed buttons:
    actions = make_actions(len(Action))

    # Start playing
    with tf.device(device):
        # Everything is executed in the context of the device (on GPU if available or on CPU).
        # Initialize the agent.
        agent = select_agent(args, actions)
        # Instantiate the ICM (Curiosity model)
        curiosity_model = ICM(len(Action))
        # Check if we need to load the weights for the agent and for the ICM
        if args.load_weights:
            load_weights(curiosity_model, agent)
        # Play the game training the agents or evaluating the loaded weights.
        play_game(game, agent, actions, curiosity_model, 
            train=(args.task == 'train' or args.task == 'pretrain'), 
            save_weights=args.save_weights, start_episode=args.start_episode,
            task=args.task)
        # At the end, close the game
        game.close()
