# CartPole LSTM Reinforcement Learning with Gymnasium Wrappers

A reinforcement learning experiment using LSTM neural networks to control a CartPole-v1 environment from OpenAI Gymnasium, enhanced with modern Gymnasium wrappers for better performance.

## Features

- LSTM-based neural network controller
- **CartPole-v1** environment (updated from v0)
- **Gymnasium Wrappers Integration**:
  - `FrameStackObservation`: Automatic temporal frame stacking
  - `NormalizeObservation`: Automatic observation normalization
- Real-time online learning during episodes
- Command-line interface with visual and text-only modes
- Configurable sequence length for temporal learning
- Comprehensive statistics reporting

## Key Improvements

- **Modern Gymnasium API**: Uses CartPole-v1 with latest Gymnasium features
- **Built-in Frame Stacking**: Replaced custom sequence management with `FrameStackObservation` wrapper
- **Automatic Normalization**: Built-in observation normalization for stable learning
- **Simplified Architecture**: Cleaner code leveraging Gymnasium's wrapper ecosystem
- **Better Performance**: Normalized inputs and proper frame stacking improve learning stability

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
- **Environment**: CartPole-v1 with Gymnasium wrappers
- **Input**: Stacked and normalized observations + action history via `FrameStackObservation` and `NormalizeObservation`
- **Architecture**: Input(sequence_length × 4 + sequence_length) → Reshape → LSTM(10 units) → Tanh → Dense(1) → Sigmoid
- **Sequence Structure**: 
  - Observations: Automatic frame stacking provides temporal context
  - Actions: Manual action history management for the last `sequence_length` actions
  - Combined input: Each timestep contains [observation_features, previous_action]
- **Sequence Length**: Configurable (default: 5), affects both observation and action history
- **Normalization**: Automatic observation normalization for stable learning
- **Training**: Online learning with MSE loss using current observations and action history
- **Action Selection**: Sigmoid output > 0.5 → action 1, else action 0
- **Memory Management**: Action history is reset between episodes

### Key Implementation Features

- **Correct Training Logic**: Uses current observations for learning, not previous ones
- **Action History Integration**: LSTM receives both stacked observations and action sequences
- **Temporal Learning**: Can learn from patterns in both state transitions and action sequences
- **Episode Management**: Action history is properly reset between episodes
- **Verbose Debugging**: Shows observation frames, action history, and input construction

## Dependencies

- gymnasium: Modern OpenAI Gym environment
- tensorflow/keras: Deep learning framework
- pygame: Visual rendering
- numpy: Numerical computations
- click: Command-line interface

## License

This project is based on the original work by Giuseppe Bonaccorso and has been updated for modern Python and Gymnasium compatibility.
