# CartPole LSTM Reinforcement Learning

A reinforcement learning experiment using LSTM neural networks to control a CartPole environment from OpenAI Gymnasium.

## Features

- LSTM-based neural network controller
- Real-time online learning during episodes
- Command-line interface with visual and text-only modes
- Configurable number of episodes and time limits
- Comprehensive statistics reporting

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Cartpole-LSTM
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Text-only Mode (Default)
Run the experiment without visual rendering for faster execution:

```bash
# Default: 100 episodes in text mode
python cartpole-lstm.py

# Custom number of episodes
python cartpole-lstm.py --episodes 50

# With shorter time limit per episode
python cartpole-lstm.py --episodes 20 --max-time 60
```

### Visual Mode
Run with pygame visual rendering to see the CartPole in action:

```bash
# Enable visual rendering
python cartpole-lstm.py --visual

# Visual mode with custom settings
python cartpole-lstm.py -v -e 25 -t 90
```

### Command Line Options

```
Options:
  -v, --visual                   Enable visual rendering with pygame (default: text-only)
  -e, --episodes INTEGER         Number of episodes to run (default: 100)
  -t, --max-time INTEGER         Maximum time per episode in seconds (default: 120)
  -s, --sequence-length INTEGER  Length of observation/action sequence for LSTM (default: 5)
  --verbose                      Enable verbose output
  --help                         Show help message and exit
```

### Examples

```bash
# Quick test with 5 episodes in text mode
python cartpole-lstm.py -e 5

# Visual training session with 50 episodes
python cartpole-lstm.py --visual --episodes 50

# Long training run with 200 episodes and sequence length of 10
python cartpole-lstm.py -e 200 -t 180 -s 10

# Test different sequence lengths
python cartpole-lstm.py -e 20 -s 3   # Short sequence
python cartpole-lstm.py -e 20 -s 15  # Long sequence

# Debug mode with verbose output
python cartpole-lstm.py --verbose --episodes 10 -s 8
```

## Output

### Text-only Mode
Shows a compact progress display:
```
Episode   1/100: Reward =   45.0, Steps =  46, Time =   1.9s
Episode   2/100: Reward =   23.0, Steps =  24, Time =   1.0s
...
```

### Visual Mode
Opens a pygame window showing the CartPole environment and prints detailed episode information.

### Final Statistics
Both modes display comprehensive results:
```
--------------------------------------------------
Final Results:
Average reward: 62.80
Std deviation: 10.72
Min reward: 46.0
Max reward: 76.0
Total episodes: 100
```

## Algorithm Details

The LSTM controller uses:
- **Input**: Sequences of CartPole states and actions for temporal learning
- **Architecture**: Input(sequence_length, 5) → LSTM(10 units) → Tanh → Dense(1) → Sigmoid
- **Sequence Structure**: Each timestep contains [position, velocity, angle, angular_velocity, previous_action]
- **Sequence Length**: Configurable (default: 5), longer sequences provide more temporal context
- **Training**: Online learning with MSE loss during each episode
- **Action Selection**: Sigmoid output > 0.5 → action 1, else action 0
- **Memory Management**: Automatic history buffer management, reset between episodes

### Sequence Learning Benefits

- **Temporal Context**: LSTM can learn from patterns across multiple timesteps
- **Memory**: Better handling of momentum and trajectory information
- **Adaptability**: Configurable sequence length for different learning strategies
- **History Management**: Automatic padding and buffer management for variable episode lengths

## Dependencies

- gymnasium: Modern OpenAI Gym environment
- tensorflow/keras: Deep learning framework
- pygame: Visual rendering
- numpy: Numerical computations
- click: Command-line interface

## License

This project is based on the original work by Giuseppe Bonaccorso and has been updated for modern Python and Gymnasium compatibility.
