# Vizdoom
from gc import collect
import vizdoom as vzd
# Common imports
import os
from typing import Dict, List, Sequence, Tuple, Union
from collections import deque
import time
import math
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# Our modules
from agent import Agent
from actor_critic_agent import BaselineActorCriticAgent
from curiosity import RND
from state import RNDScreenPreprocessor, State
from action import Action
# Variables
from variables import *

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
    args.add_argument('-ni', '--no_intrinsic', action="store_true",
        help="Train the agent without intrinsic rewards")
    return args.parse_args()

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

# Obtain the list of available actions as one-hot-encoded buttons
def make_actions(num_available_actions):
    actions = []
    for i in range(num_available_actions):
        ll = [False]*num_available_actions
        ll[i] = True
        actions.append(ll)
    return actions

def select_agent(args, actions):
    agent = args.agent
    if agent == 'random':
        return Agent(len(actions),
            tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=CLIP_NO))
    if agent == 'actor_critic':
        return BaselineActorCriticAgent(len(actions), 
            tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=CLIP_NO))
    # If we arrive here we have chosen something not implemented
    raise NotImplementedError

def load_weights(intrinsic_model, agent, task, game_map):
    try:
        intrinsic_model.load_weights(RND_WEIGHTS_PATH.format(task, game_map))
        print("Loaded weights for the RND")
    except:
        print(f"Could not find weights for the RND model at {RND_WEIGHTS_PATH.format(task, game_map)}")
    # Select actor weights based on the type of the actor
    if isinstance(agent, BaselineActorCriticAgent):
        if intrinsic_model is not None:
            try:
                agent.load_weights(ACTOR_CRITIC_WEIGHTS_PATH_RND.format(task, game_map))
                print("Loaded weights for the Actor Critic agent")
            except:
                print(f"Could not find weights for the Actor Critic model at {ACTOR_CRITIC_WEIGHTS_PATH_RND.format(task, game_map)}")
        else:
            try:
                agent.load_weights(ACTOR_CRITIC_WEIGHTS_PATH_NO_RND.format(task, game_map))
                print("Loaded weights for the Actor Critic agent")
            except:
                print(f"Could not find weights for the Actor Critic model at {ACTOR_CRITIC_WEIGHTS_PATH_NO_RND.format(task, game_map)}")
    else:
        print("Agent not implemented or does not require weights.")

def do_save_weights_and_logs(intrinsic_model:RND, agent:Agent, extrinsic_rewards:Sequence, 
                             episode:int, task:str, game_map:str):
    if intrinsic_model is not None:
        intrinsic_model.save_weights(RND_WEIGHTS_PATH.format(task, game_map))
        print(f"RND weights saved succesfully at {RND_WEIGHTS_PATH.format(task, game_map)}")
        agent.save_weights(ACTOR_CRITIC_WEIGHTS_PATH_RND.format(task, game_map))
        print(f"Agent weights saved successfully at {ACTOR_CRITIC_WEIGHTS_PATH_RND.format(task, game_map)}")
        np.save(LOGS_DIR.format(task, game_map) + '_rnd', np.array(extrinsic_rewards), allow_pickle=True)
        print(f"Rewards history saved successfully at {LOGS_DIR.format(task, game_map) + '_rnd.npy'}")
    else:
        agent.save_weights(ACTOR_CRITIC_WEIGHTS_PATH_NO_RND.format(task, game_map))
        print(f"Agent weights saved successfully at {ACTOR_CRITIC_WEIGHTS_PATH_NO_RND.format(task, game_map)}")
        np.save(LOGS_DIR.format(task, game_map) + '_no_rnd', np.array(extrinsic_rewards), allow_pickle=True)
        print(f"Rewards history saved successfully at {LOGS_DIR.format(task, game_map + '_no_rnd.npy')}")

def choose_episodes(task):
    if task == 'train':
        return TRAINING_EPISODES
    elif task == 'pretrain':
        return PRETRAINING_EPISODES
    else:
        return TESTING_EPISODES

def get_next_screen(default_screen_shape):
    # Check if we have reached the ending state
    done = game.is_episode_finished()
    if done:
        # Final state reached: create a simple black image as next image 
        # (must be uint8 for PIL to work)
        next_screen = np.zeros(default_screen_shape, dtype=np.uint8)
    else:
        # Get next image from the game
        next_screen = game.get_state().screen_buffer
    return done, next_screen

def test_step(game:vzd.DoomGame, agent:Agent, actions:List, 
            screen_preprocessor:RNDScreenPreprocessor) -> Tuple[bool, Dict]:
    # Obtain the state from the game (the image on screen)
    screen = game.get_state().screen_buffer
    # Append the frame to screen preprocessor
    screen_preprocessor.append_new_screen(screen)
    # Obtain preprocessed state for agent
    state = screen_preprocessor.get_state_for_policy_net()
    # Let the agent choose the action from the State
    action = agent.choose_action(state, training=False)['action']
    # Apply the action on the game and get the extrinsic reward.
    extrinsic_reward = game.make_action(actions[action.value], SKIP_FRAMES)
    done = game.is_episode_finished()
    return done, {
        'extrinsic_reward': extrinsic_reward,
    }

def train_step(game:vzd.DoomGame, agent:Agent, actions:List, 
            intrinsic_model:Union[RND,None], 
            screen_preprocessor:RNDScreenPreprocessor,
            iteration:int=1):
    # Obtain the state from the game (the image on screen)
    screen = game.get_state().screen_buffer
    # Append the frame to screen preprocessor
    screen_preprocessor.append_new_screen(screen)
    # Get state from preprocessor
    policy_state = State(screen_preprocessor.get_state_for_policy_net())
    # Open gradient tape to record neural network operations and compute gradients later
    with tf.GradientTape(persistent=True) as tape:
        # Let the agent choose the action from the State. This operation is recorded in the tape.
        action = agent.choose_action(policy_state, training=True, lstm_active=False)
        # Stop recording operations to compute the extrinsic reward and get the next state.
        with tape.stop_recording():
            # Apply the action on the game and get the extrinsic reward.
            extrinsic_reward = game.make_action(actions[action['action'].value], SKIP_FRAMES)
            done, next_screen = get_next_screen(screen.shape)
            # We don't append the next screen to the buffers yet, we just need to process it
            policy_next_state = State(screen_preprocessor.get_virtual_state_for_policy_net(next_screen))
            if intrinsic_model is not None:
                # Also, preprocess the next screen and pass it to the RND for computing the intrinsic reward
                next_state_rnd = screen_preprocessor.preprocess_for_rnd(next_screen)
        if intrinsic_model is not None:
            # Record on tape the next forward passes into the intrinsic model and the calculation of the intrinsic reward.
            gt_state_rnd, pred_state_rnd = intrinsic_model(tf.expand_dims(next_state_rnd, axis=0))
            # Intrinsic reward is sum of squared differences betweeen the ground truth from the untrained
            # network and the predictor's output.
            intrinsic_reward = tf.reduce_sum(tf.math.pow(gt_state_rnd - pred_state_rnd, 2))
        else:
            intrinsic_reward = tf.constant(0.0)
        # Sum rewards
        total_reward = intrinsic_reward + tf.clip_by_value(extrinsic_reward, -1.0, 1.0)
        # Compute the loss for the agent
        agent_loss = agent.compute_loss(policy_state, action, policy_next_state, total_reward, 
            {}, done, iteration, tape, lstm_active=False)
        if intrinsic_model is not None:
            # Collect the loss for the RND module (MSE)
            rnd_loss = tf.reduce_sum(intrinsic_model.losses)
        
    # Compute gradients and updates
    if intrinsic_model is not None:
        gradients_intrinsic = tape.gradient(rnd_loss, intrinsic_model.trainable_weights)    
        intrinsic_model.optimizer.apply_gradients(zip(gradients_intrinsic, intrinsic_model.trainable_weights))
    gradients_agent = tape.gradient(agent_loss, agent.trainable_weights)
    agent.optimizer.apply_gradients(zip(gradients_agent, agent.trainable_weights))
    # Remove the tape
    del tape
    return done, {      # Return if episode is done and some statistics about the step
        'extrinsic_reward': extrinsic_reward, 
        'total_reward'    : total_reward.numpy(),
        'agent_loss'      : agent_loss.numpy(),
        'intrinsic_reward': intrinsic_reward if intrinsic_model is not None else None, 
        'intrinsic_loss'  : rnd_loss.numpy() if intrinsic_model is not None else None
    }


def play_game(game:vzd.DoomGame, agent:Agent, actions:List, intrinsic_model:RND, train:bool=True,
              save_weights:bool=False, start_episode:int=0, task:str='train', game_map:str='dense'):
    global_timestep = 0
    # Create some deques for statistics about the game
    extrinsic_rewards = deque([])
    # Select maximum number of epochs based on task
    tot_episodes = choose_episodes(task)
    # Instantiate screen preprocessor
    screen_preprocessor = RNDScreenPreprocessor()
    # Start counting the playing time
    time_start = time.time()
    # Loop over episodes
    for ep in range(start_episode, tot_episodes):
        print(f"----------------------\nEpoch: {ep}/{tot_episodes-1}\n----------------------")
        screen_preprocessor.clear_preprocessed_screen_buffer()
        # Start new episode of the game
        game.new_episode()
        done = game.is_episode_finished()
        ep_timestep = 0
        with tqdm(total=math.ceil(TIMESTEPS_PER_EPISODE/SKIP_FRAMES)) as pbar:
            while not done:
                if not train:
                    done, stats = test_step(game, agent, actions, screen_preprocessor)
                else:
                    done, stats = train_step(game, agent, actions, intrinsic_model, screen_preprocessor, ep_timestep)
                ep_timestep += 1
                global_timestep += 1
                pbar.update(1)
                pbar.set_description(f"Global timestep: {global_timestep}: " + str({el: f'{stats[el]:.5f}' for el in stats}))

        # End of episode: compute aggregated statistics
        total_extrinsic_reward = game.get_total_reward()
        extrinsic_rewards.append(total_extrinsic_reward)
        print(f"Extrinsic reward: {total_extrinsic_reward:.4f}")

        if save_weights and (((ep % WEIGHTS_SAVE_FREQUENCY) == 0) or (ep == (tot_episodes-1))):
            do_save_weights_and_logs(intrinsic_model, agent, extrinsic_rewards, ep, task, game_map)

    # End of game
    time_end = time.time()
    print(f"Task finished. It took {time_end-time_start:.4f} seconds.")


if __name__ == '__main__':
    # Collect args
    args = parse_args()

    # Create folders for models and logs
    os.makedirs(os.path.join('models', 'ac_rnd'), exist_ok=True)
    os.makedirs(os.path.join('models', 'ac_no_rnd'), exist_ok=True)
    os.makedirs(os.path.join('models', 'RND'), exist_ok=True)
    os.makedirs(os.path.join('models', 'logs_rnd'), exist_ok=True)

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
        # Instantiate the RND (Curiosity model)
        intrinsic_model =   RND(tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=CLIP_NO)) \
                            if not args.no_intrinsic else None
        # Check if we need to load the weights for the agent and for the RND
        if args.load_weights:
            load_weights(intrinsic_model, agent, args.task, args.map)
        # Play the game training the agents or evaluating the loaded weights.
        play_game(game, agent, actions, intrinsic_model, 
            train=(args.task == 'train' or args.task == 'pretrain'), 
            save_weights=args.save_weights, start_episode=args.start_episode,
            task=args.task, game_map=args.map)
        # At the end, close the game
        game.close()