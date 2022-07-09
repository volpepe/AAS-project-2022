# Vizdoom
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
from reinforce_agent import BaselineREINFORCEAgent
from state import StateManager
from action import Action
from curiosity import ICM
# Variables
from variables import *

##############       ARGS MANAGEMENT         ##############
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--task', default='train', choices=['train','pretrain','test'], type=str,
        help='What task to start (training, pretraining or test). Default is "train"')
    args.add_argument('-m', '--map', default='dense', choices=['dense','sparse','verysparse'], type=str,
        help='What map to run (dense, sparse or verysparse). Pretrain map is only available using the pretrain argument in --task. Default is "dense"')
    args.add_argument('-a', '--agent', default='random', choices=['random','reinforce'], type=str,
        help='The type of agent to use (random or reinforce). Default is random.')
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

##############           UTILITIES           ##############
def choose_episodes(task):
    if task == 'train':
        return TRAINING_EPISODES
    elif task == 'pretrain':
        return PRETRAINING_EPISODES
    else:
        return TESTING_EPISODES

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
        return Agent(len(actions),
            tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=CLIP_NO))
    if agent == 'reinforce':
        return BaselineREINFORCEAgent(len(actions), 
            tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=CLIP_NO))
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

def get_next_state(state_manager:StateManager, prev_screen):
    # Check if we have reached the ending state
    done = game.is_episode_finished()
    if done:
        # Final state reached: create a simple black image as next image 
        # (must be uint8 for PIL to work)
        next_screen = np.zeros(prev_screen.shape, dtype=np.uint8)
    else:
        # Get next image from the game
        next_screen = game.get_state().screen_buffer
    # Create the next state without updating the state manager buffer
    next_state = state_manager.get_virtual_state(next_screen)
    return done, next_state

##############           TRAINING           ##############
def do_save_weights_and_logs(curiosity_model:ICM, agent:Agent, extrinsic_rewards:Sequence, 
                             episode:int, task:str, game_map:str):
    if curiosity_model is not None:
        curiosity_model.save_weights(ICM_WEIGHTS_PATH.format(task, game_map))
        print(f"ICM weights saved succesfully at {ICM_WEIGHTS_PATH.format(task, game_map)}")
        agent.save_weights(REINFORCE_WEIGHTS_PATH.format(task, game_map))
        print(f"Agent weights saved successfully at {REINFORCE_WEIGHTS_PATH.format(task, game_map)}")
        np.save(LOGS_DIR.format(task, game_map), np.array(extrinsic_rewards), allow_pickle=True)
        print(f"Rewards history saved successfully at {LOGS_DIR.format(task, game_map)}")
    else:
        agent.save_weights(REINFORCE_WEIGHTS_PATH_NO_ICM.format(task, game_map))
        print(f"Agent weights saved successfully at {REINFORCE_WEIGHTS_PATH_NO_ICM.format(task, game_map)}")
        np.save(LOGS_DIR.format(task, game_map + '_no_icm'), np.array(extrinsic_rewards), allow_pickle=True)
        print(f"Rewards history saved successfully at {LOGS_DIR.format(task, game_map + '_no_icm')}")


def load_weights(curiosity_model, agent, task, game_map):
    try:
        curiosity_model.load_weights(ICM_WEIGHTS_PATH.format(task, game_map))
        print("Loaded weights for the ICM")
    except:
        print(f"Could not find weights for the ICM model at {ICM_WEIGHTS_PATH.format(task, game_map)}")
    # Select actor weights based on the type of the actor
    if isinstance(agent, BaselineREINFORCEAgent):
        if curiosity_model is not None:
            try:
                agent.load_weights(REINFORCE_WEIGHTS_PATH.format(task, game_map))
                print("Loaded weights for the Actor Critic agent")
            except:
                print(f"Could not find weights for the Actor Critic model at {REINFORCE_WEIGHTS_PATH.format(task, game_map)}")
        else:
            try:
                agent.load_weights(REINFORCE_WEIGHTS_PATH_NO_ICM.format(task, game_map))
                print("Loaded weights for the Actor Critic agent")
            except:
                print(f"Could not find weights for the Actor Critic model at {REINFORCE_WEIGHTS_PATH_NO_ICM.format(task, game_map)}")
    else:
        print("Agent not implemented or does not require weights.")

##############           PLAYING THE GAME           ##############
def play_one_step(game:vzd.DoomGame, agent:Agent, curiosity_model:Union[ICM, None],
        action_list:List, state_manager:StateManager, training=False) -> Tuple[Dict, Dict, bool]:
    '''
    Plays one step of the game and returns the action of the model, a dictionary describing the rewards,
    and whether the episode is finished or not.
    '''
    # Obtain the state from the game (the image on screen)
    screen = game.get_state().screen_buffer
    # Update the StateManager to obtain the current state
    state = state_manager.get_current_state(screen)
    # Let the agent choose the action from the State. Also obtain the probability distribution
    # of the actions and the value from the critic
    action_info = agent.choose_action(state, training=training)
    # Unpack the action
    action = action_info['action']
    # Apply the action on the game and get the extrinsic reward.
    extrinsic_reward = game.make_action(action_list[action.value], SKIP_FRAMES)
    done, next_state = get_next_state(state_manager, screen)
    # Obtain the intrinsic reward as well (if we are using the ICM)
    if curiosity_model:
        # Inputs to the model are three tensors of batch size 1 (thus the expand_dims):
        # - State St
        # - One-hot encoded action
        # - State St+1 
        intrinsic_reward = curiosity_model(
            (tf.cast(tf.expand_dims(state.repr, axis=0           ), tf.float32), 
             tf.cast(tf.expand_dims(action_list[action.value], axis=0), tf.float32), 
             tf.cast(tf.expand_dims(next_state.repr, axis=0      ), tf.float32)),
             training=training
        )
    else:
        intrinsic_reward = 0
    total_reward = extrinsic_reward + intrinsic_reward
    # Return state, action, reward, next state, done
    return action_info, {
        'total_reward': total_reward,
        'extrinsic_reward': extrinsic_reward,
        'intrinsic_reward': intrinsic_reward
    }, done


def play_episode(game:vzd.DoomGame, agent:Agent, curiosity_model:Union[ICM,None], 
        action_list:List, state_manager:StateManager, global_timestep:int, training=False):
    '''
    Plays an episode of the game with the selected agent and optionally using the curiosity
    model for generating intrinsic rewards.
    Returns:
    - The number of steps run in the episode
    - The array of actions made in the episode
    - The rewards obtained in the episode (both from the environment and from the curiosity module)
    - The standardised cumulative rewards for each step
    '''
    # Collect all the experience played during this episode in a buffer
    episode_steps = 0
    max_steps = math.ceil(TIMESTEPS_PER_EPISODE/SKIP_FRAMES)
    episode_buffer = deque([], maxlen=max_steps)
    if training and curiosity_model is not None:
        # Keep a variable for the intrinsic loss because the add_loss API we're using resets
        # the loss at each forward pass
        intrinsic_loss = tf.constant(0.0)
    # Iterate over episode steps
    done = game.is_episode_finished()
    with tqdm(tf.range(max_steps)) as pbar:
        while not done:
            episode_steps += 1
            pbar.update()
            # Get all the necessary elements playing one step
            action_info, reward_info, done = play_one_step(
                game, agent, curiosity_model, action_list, state_manager, training=training)
            pbar.set_description(f'Global timestep: {global_timestep+episode_steps}, Intrinsic reward: {reward_info["intrinsic_reward"].numpy():.6f}')
            # Append all info in the buffer
            episode_buffer.append((action_info, reward_info))
            if training and curiosity_model is not None:
                intrinsic_loss += tf.reduce_sum(curiosity_model.losses)
            # If episode is over, break the iteration
            if done:
                break
    # Gather the experience into numpy arrays
    actions, rewards = (
        np.array([experience[i] for experience in list(episode_buffer)]) 
        for i in range(2))
    # Compute the cumulative rewards for each time step. 
    # We have to iterate backwards in the reward array and 
    # sum the rewards to a discounted running sum.
    cumulative_rewards = []
    running_sum = tf.constant(0.0)
    for i in tf.range(episode_steps-1, -1, -1):
        running_sum = rewards[i]['total_reward'] + GAMMA * running_sum
        cumulative_rewards.append(running_sum)
    # We need to flip back the array of rewards, because it was computed backwards
    cumulative_rewards = tf.stack(cumulative_rewards[::-1])
    # In some examples the cumulative rewards of the episode are standardised in order to 
    # make the training process more stable. We do that as well, since we have gathered all
    # of them.
    cumulative_rewards = (cumulative_rewards - tf.math.reduce_mean(cumulative_rewards)) / tf.math.reduce_std(cumulative_rewards)
    return episode_steps, actions, rewards, cumulative_rewards, (intrinsic_loss if training else tf.constant(0.0))


def train_step(game:vzd.DoomGame, agent:Agent, curiosity_model:Union[ICM,None], 
        action_list:List, state_manager:StateManager, global_timestep:int):
    '''
    Updates the weight of the networks after having played an episode.
    '''
    with tf.GradientTape(persistent=True) as tape:
        episode_steps, actions, rewards, cumulative_rewards, intrinsic_loss = play_episode(
            game, agent, curiosity_model, action_list, state_manager, global_timestep, training=True 
        )
        # Compute the loss for the agent
        # We need two parts:
        # 1) We call delta the difference between the computed cumulative reward and the prediction of 
        #    the value of the current state.
        #  
        #    `delta = cumulative_reward - V_pred(St)`
        #
        # 2) We compute the update for the critic using a DQN-like approach.
        # 
        #    `huber_loss(cumulative_reward, V_pred(St))`   (or `MSE`).
        #    
        # 3) Then we compute the loss for the actor:
        #
        #    `-(delta*log(action_probs))`
        #
        # 4) Finally, we compute the update for the ICM. The ICM has already computed
        #    its losses in its forward passes while playing the episode so we simply 
        #    need to retrieve them.
        
        # Collect values and log probabilities of actions for the episode
        v_st_pred = tf.squeeze(tf.stack([a['value'] for a in actions]))
        a_probs = tf.stack([a['policy'] for a in actions])
        a_indices = tf.stack([a['action'].value for a in actions])
        a_probs = tf.gather(a_probs, a_indices, batch_dims=1)
        a_log_probs = tf.math.log(a_probs)
        # Compute delta
        delta = cumulative_rewards - v_st_pred
        # Critic loss
        critic_loss = tf.keras.losses.Huber()(
            v_st_pred, cumulative_rewards)
        # Actor loss
        actor_loss = -tf.reduce_sum(tf.expand_dims(delta, axis=-1)*a_log_probs)
        # Total agent loss
        agent_loss = tf.reduce_sum(actor_loss + critic_loss)
        # ICM loss is `intrinsic_loss`

    # Update the critic and optionally the ICM
    if curiosity_model is not None:
        gradients_intrinsic = tape.gradient(intrinsic_loss, curiosity_model.trainable_weights)    
        curiosity_model.optimizer.apply_gradients(zip(gradients_intrinsic, curiosity_model.trainable_weights))
    gradients_agent = tape.gradient(agent_loss, agent.trainable_weights)    
    agent.optimizer.apply_gradients(zip(gradients_agent, agent.trainable_weights))

    # After this update, we return the episode steps and list of rewards
    return episode_steps, rewards


def play_game(game:vzd.DoomGame, agent:Agent, action_list:List, curiosity_model:ICM, train:bool=True,
              save_weights:bool=False, start_episode:int=0, task:str='train', game_map:str='dense'):
    global_timestep = 0
    # Create lists for statistics about the game
    extrinsic_rewards = deque([])       # TODO: Do we want to track something else?
    # Select maximum number of epochs based on task
    tot_episodes = choose_episodes(task)
    # Start counting the playing time
    time_start = time.time()
    # Loop over episodes
    for ep in range(start_episode, tot_episodes):
        print(f"----------------------\nEpisode: {ep}/{tot_episodes-1}\n----------------------")
        # Creates a new StateManager handling preprocessing of images and the screen buffer
        # for stacking frames into a single state
        state_manager = StateManager()
        # Start new episode of the game
        game.new_episode()
        if train:
            episode_steps, rewards_info = train_step(game, agent, curiosity_model, 
                action_list, state_manager, global_timestep)
        else:
            # Run the episode until it's over.
            episode_steps, _, rewards_info, _, _ = play_episode(
                game, agent, curiosity_model, action_list, state_manager, global_timestep)
        # Update global timestep counter
        global_timestep += episode_steps
        # End of episode: collect reward into the extrinsic rewards list
        total_extrinsic_reward = game.get_total_reward()
        extrinsic_rewards.append({global_timestep: total_extrinsic_reward})      

        # Save weights (once every WEIGHTS_SAVE_FREQUENCY episodes)
        if save_weights and (((ep % WEIGHTS_SAVE_FREQUENCY) == 0) or (ep == (tot_episodes-1))):
            do_save_weights_and_logs(curiosity_model, agent, extrinsic_rewards, ep, task, game_map)

    # End of game
    time_end = time.time()
    print(f"Task finished. It took {time_end-time_start:.4f} seconds.")

##################### START #####################
if __name__ == '__main__':
    # Collect args
    args = parse_args()

    # Create folders for models and logs
    os.makedirs(os.path.join('models', 'ac'), exist_ok=True)
    os.makedirs(os.path.join('models', 'ac_no_icm'), exist_ok=True)
    os.makedirs(os.path.join('models', 'ICM'), exist_ok=True)
    os.makedirs(os.path.join('models', 'logs'), exist_ok=True)

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
        curiosity_model =   ICM(len(Action), 
                                tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=CLIP_NO)) \
                            if not args.no_intrinsic else None
        # Check if we need to load the weights for the agent and for the ICM
        if args.load_weights:
            load_weights(curiosity_model, agent, args.task, args.map)
        # Play the game training the agents or evaluating the loaded weights.
        play_game(game, agent, actions, curiosity_model, 
            train=(args.task == 'train' or args.task == 'pretrain'), 
            save_weights=args.save_weights, start_episode=args.start_episode,
            task=args.task, game_map=args.map)
        # At the end, close the game
        game.close()
