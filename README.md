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

### 🎨 Advanced Visualization Features (New!)

- **Multi-Page Web Dashboard**: Comprehensive monitoring interface with three specialized pages
  - **Training Dashboard**: Real-time training metrics and performance
    - Side-by-side current maze frame and exploration heatmap
    - Episode rewards with smoothing
    - Episode length tracking
    - Training loss monitoring
    - Success rate tracking
    - Live visualization updates every 10 steps
    - Training statistics overview
  - **Demo Dashboard**: Demo performance analysis and visualization
    - Real-time demo frame visualization
    - Performance summary charts (steps, rewards, success rate)
    - Recorded episode management
    - Demo command reference
  - **Episode Viewer**: Interactive episode playback
    - Frame-by-frame analysis
    - Action and reward annotations
    - Interactive HTML viewer embedding

- **Demo Visualization System**: Automatic performance tracking
  - Real-time frame saving during demo execution
  - Automatic summary generation with 4-panel charts
  - Episode recording for detailed analysis
  - Headless environment support

- **Real-Time Heatmap Visualization**: Track agent exploration patterns during training
  - Live heatmap updates every 10 steps (synchronized with maze rendering)
  - Visualize most-visited maze positions
  - Episode path tracking and cumulative exploration
  - Coverage analysis
  - Compare exploration across episodes
  - Side-by-side display with current maze state in dashboard

- **Episode Replay System**: Record and replay agent episodes
  - Save episodes during training or demo
  - Frame-by-frame playback with annotations
  - Export episodes as video files
  - Compare performance across training stages

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
- Early stopping when target success rate achieved (default: 95%)

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- OpenCV (headless version for compatibility)
- Matplotlib
- **Streamlit** (for web dashboard)
- **Plotly** (for interactive plots)
- **Pandas** (for data management)

**Note**: This project uses `opencv-python-headless` for compatibility with headless environments (containers, codespaces, servers). Visual rendering in demo mode may not work in headless environments, but training works perfectly. The web dashboard works great in all environments!

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

# With visual rendering and live visualization
python main.py train --render --live-viz --episodes 1000

# All training options
python main.py train --help
```

**Training Options:**
- `--maze-size`: Size of the maze (default: 10)
- `--render-size`: Size of rendered images for training (default: 84)
- `--episodes`: Number of training episodes (default: 1000)
- `--render`: Show visual rendering during training
- `--live-viz`: Enable live training visualization (real-time charts and heatmaps)
- `--save-freq`: Frequency to save model checkpoints (default: 100)
- `--record-freq`: Frequency to record episodes (default: 50, set to 0 to disable)

**Training Features:**
- **Early Stopping**: Training automatically stops when 95% success rate is achieved (configurable)
- **Real-Time Visualization**: Live maze rendering and exploration heatmap updates every 10 steps
- **Progress Tracking**: Auto-saved training logs for dashboard monitoring
- **Model Checkpoints**: Periodic saves and final model preservation

### Demo

Visualize a trained agent solving mazes with automatic performance tracking:

```bash
# Basic demo
python demo.py --model ./models/dqn_final.pth

# Demo with visualization and episode recording
python demo.py --model ./models/dqn_final.pth --episodes 10 --record

# Demo with custom output directory
python demo.py --model ./models/dqn_final.pth --save-dir my_demo --record

# Using main.py
python main.py demo --model ./models/dqn_final.pth

# All demo options
python demo.py --help
```

**Demo Options:**
- `--model`: Path to trained model (default: ./models/dqn_final.pth)
- `--maze-size`: Size of the maze (default: 10)
- `--render-size`: Size of rendered images (default: 84) - **IMPORTANT**: Must match the render_size used during training
- `--episodes`: Number of episodes to run (default: 5)
- `--delay`: Delay between steps in ms (default: 200)
- `--save-dir`: Directory to save demo outputs (default: demo_outputs)
- `--record`: Record episodes as .pkl files for replay

**Important Note**: The `--render-size` parameter must match the size used during training (default: 84) because the neural network architecture depends on the input image dimensions. Using a different size will cause model loading errors.

**Demo Output Files:**
- `demo_current.png`: Current frame during demo execution (updates in real-time)
- `demo_summary.png`: 4-panel performance summary (steps, rewards, success rate, statistics)
- `demo_episode_0.pkl`, `demo_episode_1.pkl`, etc.: Recorded episodes (when using --record)

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

### 🎨 Visualization Features

#### Web Dashboard (Recommended)

Launch the interactive multi-page web dashboard for comprehensive monitoring:

```bash
# Start the dashboard
streamlit run src/web_dashboard.py

# Or use the wrapper script
python run_dashboard.py

# Then open your browser to: http://localhost:8501
```

**Dashboard Pages:**

1. **Training Dashboard**
   - Real-time training metrics and plots
   - Episode rewards, length, loss tracking
   - Success rate monitoring
   - Live visualization updates
   - Training statistics overview
   - Model selection and management

2. **Demo Dashboard**
   - Current demo frame visualization
   - Performance summary charts
   - Recorded episodes list
   - Demo command reference
   - Quick actions for running demos

3. **Episode Viewer**
   - Interactive HTML episode playback
   - Frame-by-frame navigation
   - Action and reward annotations
   - Episode data visualization

**Features:**
- Multi-page navigation for organized access
- Auto-refresh for Training Dashboard
- Interactive Plotly charts
- Page-specific controls and information
- Clean, isolated content per page

#### Episode Replay

View recorded episodes with frame-by-frame playback:

```bash
# View a training episode
python replay_viewer.py outputs/episode_100.pkl

# View a demo episode
python replay_viewer.py demo_outputs/demo_episode_0.pkl

# Show episode summary only
python replay_viewer.py demo_outputs/demo_episode_0.pkl --summary

# Export episode as video
python replay_viewer.py demo_outputs/demo_episode_0.pkl --export-video output/demo.mp4 --fps 15

# Export frames to generate HTML viewer
python replay_viewer.py demo_outputs/demo_episode_0.pkl --export-images frames/
```

**Replay Controls:**
- `q` - Quit
- `p` - Pause/Resume
- `n` - Next frame
- `b` - Previous frame

**Note:** Exporting frames with `--export-images` generates a `viewer.html` file that can be viewed in the Episode Viewer page of the web dashboard.

#### Heatmap Visualization

Heatmaps are automatically generated in real-time during training and saved to `outputs/`:

**Real-Time Updates:**
- `heatmap_current.png` - Updates every 10 steps during training
- Synchronized with `render_current.png` for side-by-side comparison
- Shows cumulative exploration including ongoing episode
- Displayed in Training Dashboard alongside current maze frame

**Features:**
- Most frequently visited positions (hotter colors = more visits)
- Agent exploration patterns and path tracking
- Coverage analysis and statistics
- Episode-by-episode path visualization
- Final summary: `exploration_heatmap.png`

## Project Structure

```
TestProject/
├── main.py                    # Main entry point
├── demo.py                    # Demo script with visualization
├── run_dashboard.py           # Launch web dashboard
├── replay_viewer.py           # Episode replay viewer
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── maze_environment.py    # Maze environment
│   ├── dqn_network.py         # Neural network architectures
│   ├── dqn_agent.py           # DQN agent with replay buffer
│   ├── train.py               # Training script
│   ├── live_visualization.py  # Real-time training plots
│   ├── heatmap_visualizer.py  # Exploration heatmaps
│   ├── episode_replay.py      # Episode recording & replay
│   └── web_dashboard.py       # Multi-page Streamlit dashboard
├── models/                    # Saved model checkpoints
│   ├── dqn_final.pth          # Final trained model
│   └── dqn_early_stopped_*.pth # Early-stopped models (if triggered)
├── outputs/                   # Training plots, metrics, replays
│   ├── render_current.png     # Current maze state (real-time)
│   ├── heatmap_current.png    # Current exploration heatmap (real-time)
│   ├── live_training_viz.png  # Training metrics chart
│   ├── exploration_heatmap.png # Final exploration summary
│   ├── training_log.json      # Training metrics log
│   └── episode_*.pkl          # Recorded training episodes
├── demo_outputs/              # Demo visualizations and recordings
│   ├── demo_current.png       # Current demo frame
│   ├── demo_summary.png       # Performance summary
│   └── demo_episode_*.pkl     # Recorded demo episodes
├── viewer.html                # Generated episode viewer (optional)
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

**Model loading error (size mismatch):**
- Ensure `--render-size` matches the training render size (default: 84)
- The neural network architecture depends on input image dimensions
- If you trained with custom render_size, use the same value for demo

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