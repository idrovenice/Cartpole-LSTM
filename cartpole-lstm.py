'''
OpenAI-Gym Cartpole-v1 LSTM experiment
'''

import os
import warnings

# Suppress TensorFlow logging and warnings before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA to avoid GPU-related messages
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, NormalizeObservation
import numpy as np
import time
import click

# Additional TensorFlow logging suppression
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Input, Reshape
from keras import backend as K

# Result location
result_location = './results'

# Number of episodes
nb_episodes = 100

# Max execution time (in seconds)
max_execution_time = 120

# Set random seed
np.random.seed(1000)


class CartPoleController(object):
    def __init__(self, n_input=4, n_hidden=10, n_output=1, initial_state=0.1, training_threshold=1.5, sequence_length=5):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.initial_state = initial_state
        self.training_threshold = training_threshold
        self.step_threshold = 0.5
        self.sequence_length = sequence_length
        
        # Store action history for the LSTM input
        self.action_history = []
        
        # With FrameStack wrapper, we'll receive stacked observations directly
        # Input shape is (sequence_length * n_input + sequence_length,) 
        # - stacked observations + stacked actions
        self.obs_input_dim = sequence_length * n_input
        self.action_input_dim = sequence_length
        self.total_input_dim = self.obs_input_dim + self.action_input_dim

        # Action neural network
        # Input is flattened stacked observations + stacked actions
        # LSTM -> (n_hidden)
        # Dense output -> (n_output)
        self.action_model = Sequential()
        
        # Reshape flattened input back to sequence format for LSTM
        # Each timestep has (n_input + 1) features: observation + previous action
        self.action_model.add(Input(shape=(self.total_input_dim,)))
        self.action_model.add(Dense(self.total_input_dim, activation='linear'))  # Optional preprocessing
        self.action_model.add(Reshape((self.sequence_length, self.n_input + 1)))
        self.action_model.add(LSTM(self.n_hidden))
        self.action_model.add(Activation('tanh'))
        self.action_model.add(Dense(self.n_output))
        self.action_model.add(Activation('sigmoid'))

        self.action_model.compile(loss='mse', optimizer='adam')

    def reset_episode(self):
        """Reset the action history at the start of each episode"""
        self.action_history = []

    def _prepare_input(self, obs):
        """Prepare the input combining stacked observations and action history"""
        # obs is already stacked by FrameStack wrapper: shape (sequence_length, n_input)
        # Flatten observations
        obs_flat = obs.flatten()  # shape: (sequence_length * n_input,)
        
        # Pad action history if needed
        current_actions = len(self.action_history)
        if current_actions < self.sequence_length:
            # Pad with zeros for missing actions
            padded_actions = [0] * (self.sequence_length - current_actions) + self.action_history
        else:
            # Take last sequence_length actions
            padded_actions = self.action_history[-self.sequence_length:]
        
        # Combine observations and actions
        action_array = np.array(padded_actions, dtype=np.float32)
        combined_input = np.concatenate([obs_flat, action_array])
        
        return combined_input.reshape(1, -1).astype(K.floatx())

    def action(self, obs, prev_obs=None, prev_action=None, verbose=False):
        # Prepare input with current observations and action history
        x = self._prepare_input(obs)
        
        # Training step using previous observation if available
        if prev_obs is not None and prev_action is not None:
            prev_norm = np.linalg.norm(prev_obs.flatten())

            if prev_norm > self.training_threshold:
                # Prepare training input using previous observation and action history
                train_x = self._prepare_input(prev_obs)
                
                if prev_norm < self.step_threshold:
                    y = np.array([prev_action]).astype(K.floatx())
                else:
                    y = np.array([np.abs(prev_action - 1)]).astype(K.floatx())

                self.action_model.train_on_batch(train_x, y)

        if verbose:
            print(f"obs shape: {obs.shape}")
            print(f"obs values (first 8): {obs.flatten()[:8].round(3)}")
            print(f"action history: {self.action_history}")
            print(f"input shape to model: {x.shape}")
            # Show the stacked observations and actions
            print("Stacked observations:")
            for i in range(self.sequence_length):
                start_idx = i * self.n_input
                end_idx = (i + 1) * self.n_input
                frame_obs = obs.flatten()[start_idx:end_idx]
                action_idx = len(self.action_history) - self.sequence_length + i if len(self.action_history) >= self.sequence_length else max(0, len(self.action_history) + i - self.sequence_length + 1)
                prev_act = self.action_history[action_idx] if 0 <= action_idx < len(self.action_history) else 0
                print(f"  Frame {i+1}: obs={frame_obs.round(3)}, prev_action={prev_act}")
            print("---")
        
        # Predict new action
        output = self.action_model.predict(x, batch_size=1, verbose=0)
        new_action = self.step(output)
        
        # Store the new action in history
        self.action_history.append(new_action)
        
        return new_action

    def step(self, value):
        if value > self.step_threshold:
            return int(1)
        else:
            return int(0)


def run_experiment(visual_mode=True, episodes=100, max_time=120, sequence_length=5, verbose=False):
    """Run the CartPole LSTM experiment"""
    
    # Set random seed
    np.random.seed(1000)
    
    print('OpenAI-Gym CartPole-v1 LSTM experiment with Gymnasium Wrappers')
    print(f"Mode: {'Visual' if visual_mode else 'Text-only'}")
    print(f"Episodes: {episodes}")
    print(f"Max time per episode: {max_time}s")
    print(f"Sequence length: {sequence_length}")
    print("-" * 50)

    # Create environment with wrappers
    if visual_mode:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    # Apply wrappers
    env = NormalizeObservation(env)  # Normalize observations
    env = FrameStackObservation(env, stack_size=sequence_length)  # Stack frames for temporal context
    
    cart_pole_controller = CartPoleController(sequence_length=sequence_length)
    total_reward = []

    for episode in range(episodes):
        # Reset environment and action history
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            observation = reset_result[0]
        else:
            observation = reset_result
        
        # Reset the controller's action history
        cart_pole_controller.reset_episode()
        
        # Get first action
        action = cart_pole_controller.action(observation, verbose=verbose)
        previous_observation = observation
        previous_action = action

        done = False
        t = 0
        partial_reward = 0.0
        start_time = time.time()
        elapsed_time = 0

        while not done and elapsed_time < max_time:
            t += 1
            elapsed_time = time.time() - start_time

            if visual_mode:
                env.render()
            
            # Handle both old and new Gym API
            step_result = env.step(action)
            if len(step_result) == 5:
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                observation, reward, done, info = step_result
            partial_reward += reward

            action = cart_pole_controller.action(observation, previous_observation, previous_action, verbose=verbose)
            previous_observation = observation
            previous_action = action

        # Print episode results
        if visual_mode:
            print('Episode %d finished after %d timesteps. Total reward: %1.0f. Elapsed time: %1.1f s' %
                  (episode+1, t+1, partial_reward, elapsed_time))
        else:
            # Text-only mode: show progress and reward
            print(f"Episode {episode+1:3d}/{episodes}: Reward = {partial_reward:6.1f}, Steps = {t+1:3d}, Time = {elapsed_time:5.1f}s")

        total_reward.append(partial_reward)

    env.close()
    
    # Final statistics
    total_reward = np.array(total_reward)
    print("-" * 50)
    print('Final Results:')
    print(f'Average reward: {np.mean(total_reward):.2f}')
    print(f'Std deviation: {np.std(total_reward):.2f}')
    print(f'Min reward: {np.min(total_reward):.1f}')
    print(f'Max reward: {np.max(total_reward):.1f}')
    print(f'Total episodes: {len(total_reward)}')


@click.command()
@click.option('--visual', '-v', is_flag=True, default=False, 
              help='Enable visual rendering with pygame (default: text-only)')
@click.option('--episodes', '-e', default=100, type=int,
              help='Number of episodes to run (default: 100)')
@click.option('--max-time', '-t', default=120, type=int,
              help='Maximum time per episode in seconds (default: 120)')
@click.option('--sequence-length', '-s', default=5, type=int,
              help='Length of observation/action sequence for LSTM (default: 5)')
@click.option('--verbose', is_flag=True, default=False,
              help='Enable verbose output')
def main(visual, episodes, max_time, sequence_length, verbose):
    """
    CartPole LSTM Reinforcement Learning Experiment with Gymnasium Wrappers
    
    This script trains an LSTM neural network to control a CartPole-v1 environment.
    Uses Gymnasium's FrameStack and NormalizeObservation wrappers for better performance.
    
    Examples:
    
    \b
    # Run in text mode (default)
    python cartpole-lstm.py
    
    \b
    # Run with visual rendering
    python cartpole-lstm.py --visual
    
    \b
    # Run 50 episodes with sequence length of 10
    python cartpole-lstm.py --episodes 50 --sequence-length 10
    
    \b
    # Run with visual rendering and custom settings
    python cartpole-lstm.py -v -e 50 -t 60 -s 8
    """
    
    if verbose:
        print(f"Running with visual={'on' if visual else 'off'}, episodes={episodes}, max_time={max_time}, sequence_length={sequence_length}")
    
    try:
        run_experiment(visual_mode=visual, episodes=episodes, max_time=max_time, sequence_length=sequence_length, verbose=verbose)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"Error during experiment: {e}")


if __name__ == '__main__':
    main()
