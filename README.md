# Visual Maze Solving with Deep Reinforcement Learning

A complete implementation of a Deep Q-Network (DQN) agent that learns to navigate and solve mazes using visual input. Built with Python and PyTorch.

## Features

- **Visual Maze Environment**: Procedurally generated mazes with visual rendering
- **Deep Q-Network (DQN)**: CNN-based neural network for processing visual input
- **Dueling DQN Architecture**: Enhanced Q-learning with separate value and advantage streams
- **Experience Replay**: Efficient learning from past experiences
- **Target Network**: Stabilized training with periodic target updates
- **Training Visualization**: Real-time metrics and performance plots
- **Demo Mode**: Visualize trained agents solving mazes

## Architecture

### Environment
- Procedurally generated mazes using recursive backtracking
- Visual observations (RGB images)
- 4 actions: UP, DOWN, LEFT, RIGHT
- Rewards: +100 for reaching goal, -0.1 per step, distance-based shaping

### Neural Network
- **Input**: 84x84x3 RGB images
- **Architecture**:
  - 3 Convolutional layers with batch normalization
  - 2 Fully connected layers
  - Dueling architecture (optional)
- **Output**: Q-values for each action

### Training Algorithm
- Deep Q-Learning with Experience Replay
- Epsilon-greedy exploration
- Target network for stability
- Huber loss (Smooth L1)
- Adam optimizer

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- OpenCV
- Matplotlib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd TestProject

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Train a new agent (quick training)
python main.py train --episodes 500

# Demo the trained agent
python main.py demo
```

### Training

Train a DQN agent on a maze:

```bash
# Basic training
python main.py train

# Custom maze size and more episodes
python main.py train --maze-size 15 --episodes 2000

# With visual rendering during training
python main.py train --render --episodes 1000

# All training options
python main.py train --help
```

**Training Options:**
- `--maze-size`: Size of the maze (default: 10)
- `--render-size`: Size of rendered images for training (default: 84)
- `--episodes`: Number of training episodes (default: 1000)
- `--render`: Show visual rendering during training
- `--save-freq`: Frequency to save model checkpoints (default: 100)

### Demo

Visualize a trained agent solving mazes:

```bash
# Demo with default model
python main.py demo

# Demo with specific model
python main.py demo --model ./models/dqn_checkpoint_ep500.pth

# Custom settings
python main.py demo --episodes 10 --delay 100

# All demo options
python main.py demo --help
```

**Demo Options:**
- `--model`: Path to trained model (default: ./models/dqn_final.pth)
- `--maze-size`: Size of the maze (default: 10)
- `--render-size`: Size of rendered images (default: 400)
- `--episodes`: Number of episodes to run (default: 5)
- `--delay`: Delay between steps in ms (default: 200)

**Demo Controls:**
- Press `q` to quit
- Press `n` to skip to next episode

### Advanced Usage

#### Direct Training Script

```bash
# Using the training script directly
python src/train.py --maze-size 12 --episodes 1500
```

#### Direct Demo Script

```bash
# Using the demo script directly
python demo.py --model ./models/dqn_final.pth --episodes 10
```

## Project Structure

```
TestProject/
├── main.py                 # Main entry point
├── demo.py                 # Demo script
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── maze_environment.py # Maze environment implementation
│   ├── dqn_network.py      # Neural network architectures
│   ├── dqn_agent.py        # DQN agent with replay buffer
│   └── train.py            # Training script
├── models/                 # Saved model checkpoints
├── outputs/                # Training plots and metrics
└── README.md
```

## How It Works

### 1. Maze Generation
The environment generates random mazes using recursive backtracking algorithm. Each maze has:
- Start position (blue circle)
- Goal position (green circle)
- Walls (gray) and paths (white)

### 2. Visual Input Processing
The agent receives visual observations (RGB images) of the maze. The CNN processes these images to extract spatial features.

### 3. Deep Q-Learning
The agent learns to estimate Q-values (expected future rewards) for each action using:
- **Experience Replay**: Stores transitions and samples random batches for training
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy**: Balances exploration and exploitation

### 4. Training Process
1. Agent explores the maze using epsilon-greedy policy
2. Experiences are stored in replay buffer
3. Random batches are sampled for training
4. Network learns to predict optimal Q-values
5. Epsilon gradually decreases (more exploitation over time)

### 5. Evaluation
Success is measured by:
- Reaching the goal position
- Number of steps taken
- Total reward accumulated

## Training Results

After training, you'll find:
- **Model checkpoints**: Saved in `./models/`
- **Training metrics**: Plot saved in `./outputs/training_metrics.png`
  - Episode rewards over time
  - Episode length (efficiency)
  - Training loss
  - Success rate

## Hyperparameters

Default hyperparameters (can be modified in `src/dqn_agent.py`):

```python
learning_rate = 1e-4        # Learning rate for Adam optimizer
gamma = 0.99                # Discount factor
epsilon_start = 1.0         # Initial exploration rate
epsilon_end = 0.01          # Final exploration rate
epsilon_decay = 0.995       # Epsilon decay per episode
buffer_capacity = 10000     # Replay buffer size
batch_size = 32             # Training batch size
target_update_freq = 500    # Target network update frequency
```

## Performance Tips

1. **Faster Training**: Use smaller mazes (--maze-size 8) for quicker convergence
2. **Better Performance**: Train longer (--episodes 2000+) for more complex mazes
3. **GPU Acceleration**: PyTorch will automatically use CUDA if available
4. **Monitoring**: Use `--render` during training to watch the agent learn

## Troubleshooting

**Agent not learning:**
- Train for more episodes
- Try smaller maze first
- Check that PyTorch is installed correctly

**Slow training:**
- Reduce render_size (default 84 is optimal)
- Remove `--render` flag
- Use GPU if available

**Demo not working:**
- Ensure model exists: `ls models/`
- Train a model first: `python main.py train`
- Check model path: `--model ./models/dqn_final.pth`

## Technical Details

### Deep Q-Network Architecture

```
Input: 84x84x3 RGB Image
    ↓
Conv2D(32, 8x8, stride=4) + BatchNorm + ReLU
    ↓
Conv2D(64, 4x4, stride=2) + BatchNorm + ReLU
    ↓
Conv2D(64, 3x3, stride=1) + BatchNorm + ReLU
    ↓
Flatten
    ↓
Dense(512) + ReLU
    ↓
Dense(num_actions)
    ↓
Q-values for each action
```

### Dueling DQN Architecture

Separates value and advantage estimation:
- **Value Stream**: Estimates state value V(s)
- **Advantage Stream**: Estimates advantage A(s,a)
- **Combination**: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## Authors

Created with Deep Reinforcement Learning and PyTorch