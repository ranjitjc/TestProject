"""
Training script for DQN agent on maze environment
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.maze_environment import MazeEnvironment
from src.dqn_agent import DQNAgent
from src.episode_replay import EpisodeRecorder
from src.heatmap_visualizer import HeatmapVisualizer
from src.live_visualization import LiveTrainingVisualizer


class Trainer:
    """Trainer class for DQN agent on maze environment."""

    def __init__(
        self,
        maze_size: int = 10,
        render_size: int = 84,
        num_episodes: int = 1000,
        save_freq: int = 100,
        model_dir: str = './models',
        output_dir: str = './outputs',
        record_freq: int = 50,
        enable_live_viz: bool = False
    ):
        """
        Initialize the trainer.

        Args:
            maze_size: Size of the maze
            render_size: Size of rendered images
            num_episodes: Number of training episodes
            save_freq: Frequency to save model checkpoints
            model_dir: Directory to save models
            output_dir: Directory to save outputs
            record_freq: Frequency to record episodes (0 = disable)
            enable_live_viz: Enable live training visualization
        """
        self.maze_size = maze_size
        self.render_size = render_size
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.record_freq = record_freq
        self.enable_live_viz = enable_live_viz

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize environment
        self.env = MazeEnvironment(maze_size=maze_size, render_size=render_size)

        # Initialize agent
        input_shape = (render_size, render_size, 3)
        self.agent = DQNAgent(
            input_shape=input_shape,
            num_actions=self.env.num_actions,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_capacity=10000,
            batch_size=32,
            target_update_freq=500,
            use_dueling=True
        )

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.success_rate = []

        # Visualization tools
        self.heatmap_viz = HeatmapVisualizer(maze_size=maze_size)
        self.live_viz = LiveTrainingVisualizer(
            window_size=100,
            update_interval=10,
            output_dir=self.output_dir
        ) if enable_live_viz else None

        # Track best episode for recording
        self.best_reward = float('-inf')
        self.best_episode_data = None

    def train(self, render: bool = False):
        """
        Train the agent.

        Args:
            render: Whether to render the environment during training
        """
        print(f"Training DQN agent on {self.maze_size}x{self.maze_size} maze")
        print(f"Device: {self.agent.device}")
        print(f"Number of episodes: {self.num_episodes}")
        print(f"Episode recording: Every {self.record_freq} episodes" if self.record_freq > 0 else "Episode recording: Disabled")
        print(f"Live visualization: {'Enabled' if self.enable_live_viz else 'Disabled'}")

        # Detect headless environment
        import os
        self.is_headless = os.environ.get('DISPLAY') is None
        if self.is_headless and (render or self.enable_live_viz):
            print(f"Headless mode: Visualizations will save to {self.output_dir}/")
            print(f"  - render_current.png (current maze state)")
            print(f"  - heatmap_current.png (exploration heatmap)")
        print("-" * 50)

        recent_successes = []
        episode_path = []  # Track agent path for current episode

        for episode in tqdm(range(self.num_episodes), desc="Training"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            done = False
            episode_path = []

            # Determine if we should record this episode
            should_record = (
                self.record_freq > 0 and (
                    episode == 0 or  # First episode
                    (episode + 1) % self.record_freq == 0 or  # Every N episodes
                    episode == self.num_episodes - 1  # Last episode
                )
            )

            if should_record:
                recorder = EpisodeRecorder()

            while not done:
                # Record agent position for heatmap (both for path and real-time tracking)
                current_pos = tuple(self.env.agent_pos)
                episode_path.append(current_pos)
                self.heatmap_viz.record_position(current_pos)  # Update heatmap in real-time

                # Select and perform action
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)

                # Record step if recording this episode
                if should_record:
                    recorder.record_step(state, action, reward, info)

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train agent
                loss = self.agent.train_step()
                episode_loss += loss

                # Update state
                state = next_state
                episode_reward += reward
                steps += 1

                # Render if requested or if live visualization is enabled
                if render or self.enable_live_viz:
                    # Save heatmap alongside render every 10 steps for real-time tracking
                    if steps % 10 == 0:
                        try:
                            self.heatmap_viz.save_heatmap(
                                f'{self.output_dir}/heatmap_current.png',
                                verbose=False,  # Suppress print to avoid spam during training
                                maze=self.env.maze,
                                start_pos=self.env.start_pos,
                                goal_pos=self.env.goal_pos
                            )
                        except Exception:
                            pass  # Silently fail if heatmap save fails during training

                    if self.is_headless:
                        # Save to file in headless mode (every 10 steps to avoid too many files)
                        if steps % 10 == 0:
                            save_path = f'{self.output_dir}/render_current.png'
                            self.env.render('human', save_path=save_path)
                    else:
                        # Display in window and also save to file for dashboard
                        self.env.render('human')
                        if steps % 10 == 0:
                            save_path = f'{self.output_dir}/render_current.png'
                            self.env.render('rgb_array', save_path=save_path)
                        if not render:  # Only sleep if render flag is explicitly set
                            time.sleep(0.01)

            # Update epsilon
            self.agent.update_epsilon()

            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            if episode_loss > 0:
                self.losses.append(episode_loss / steps)

            # Track success rate
            success = self.env._is_goal_reached()
            recent_successes.append(1 if success else 0)
            if len(recent_successes) > 100:
                recent_successes.pop(0)

            success_rate = np.mean(recent_successes) * 100

            # Store episode path for later analysis (positions already recorded in real-time)
            self.heatmap_viz.episode_paths.append(episode_path)

            # Save episode recording if needed
            if should_record:
                recorder.finalize(
                    success=success,
                    total_steps=steps,
                    total_reward=episode_reward,
                    episode_num=episode
                )
                recorder.save(f'{self.output_dir}/episode_{episode}.pkl')

            # Track best episode for later saving
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                # Store best episode data
                if should_record:
                    self.best_episode_data = (episode, recorder)

            # Update live visualization
            if self.live_viz:
                self.live_viz.update(
                    episode=episode,
                    reward=episode_reward,
                    length=steps,
                    loss=episode_loss / steps if episode_loss > 0 else 0,
                    epsilon=self.agent.epsilon,
                    success=success
                )

            # Export training log for dashboard (every 10 episodes)
            if (episode + 1) % 10 == 0 or episode == self.num_episodes - 1:
                self._export_training_log()

            # Print progress
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                print(f"\nEpisode {episode + 1}/{self.num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")

            # Save model checkpoint
            if (episode + 1) % self.save_freq == 0:
                self.save_model(f"dqn_checkpoint_ep{episode + 1}.pth")

            # Update agent episode counter
            self.agent.episodes = episode + 1

        # Save final model
        self.save_model("dqn_final.pth")

        # Save visualizations
        print("\nSaving visualizations...")

        # Export final training log
        try:
            self._export_training_log()
            print(f"  ✓ Training log exported to outputs/training_log.json")
        except Exception as e:
            print(f"  ✗ Training log export failed: {e}")

        # Save heatmap
        try:
            self.heatmap_viz.save_heatmap(
                f'{self.output_dir}/exploration_heatmap.png',
                maze=self.env.maze,
                start_pos=self.env.start_pos,
                goal_pos=self.env.goal_pos
            )
            print(f"  ✓ Exploration heatmap saved")
        except Exception as e:
            print(f"  ✗ Heatmap save failed: {e}")

        # Save live visualization if enabled
        if self.live_viz:
            try:
                self.live_viz.save(f'{self.output_dir}/live_training_viz.png')
                self.live_viz.close()
                print(f"  ✓ Live visualization saved")
            except Exception as e:
                print(f"  ✗ Live visualization save failed: {e}")

        # Print heatmap statistics
        stats = self.heatmap_viz.get_statistics()
        print(f"\nExploration Statistics:")
        print(f"  Total visits: {stats['total_visits']}")
        print(f"  Unique cells: {stats['unique_cells_visited']}")
        print(f"  Coverage: {stats['coverage']:.1f}%")

        # Close environment
        self.env.close()

        print("\nTraining completed!")

    def save_model(self, filename: str):
        """Save the model."""
        filepath = os.path.join(self.model_dir, filename)
        self.agent.save(filepath)

    def _export_training_log(self):
        """Export training metrics as JSON for the web dashboard."""
        # Prepare data for export
        training_data = []

        # Calculate success rate for each episode
        for i in range(len(self.episode_rewards)):
            # Calculate rolling success rate
            episode_success = 1 if self.episode_rewards[i] > 50 else 0

            # Calculate success rate over last 100 episodes
            start_idx = max(0, i - 99)
            recent_rewards = self.episode_rewards[start_idx:i+1]
            recent_successes = [1 if r > 50 else 0 for r in recent_rewards]
            success_rate = np.mean(recent_successes) * 100

            # Get loss if available
            loss = self.losses[i] if i < len(self.losses) else 0.0

            training_data.append({
                'episode': i,
                'reward': float(self.episode_rewards[i]),
                'length': int(self.episode_lengths[i]),
                'loss': float(loss),
                'success_rate': float(success_rate)
            })

        # Save to JSON file
        log_path = os.path.join(self.output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_data, f, indent=2)

    def plot_training_metrics(self):
        """Plot and save training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3)
        if len(self.episode_rewards) > 50:
            smoothed = np.convolve(
                self.episode_rewards,
                np.ones(50) / 50,
                mode='valid'
            )
            axes[0, 0].plot(smoothed, linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True)

        # Plot episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.3)
        if len(self.episode_lengths) > 50:
            smoothed = np.convolve(
                self.episode_lengths,
                np.ones(50) / 50,
                mode='valid'
            )
            axes[0, 1].plot(smoothed, linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].grid(True)

        # Plot losses
        if self.losses:
            axes[1, 0].plot(self.losses, alpha=0.3)
            if len(self.losses) > 50:
                smoothed = np.convolve(
                    self.losses,
                    np.ones(50) / 50,
                    mode='valid'
                )
                axes[1, 0].plot(smoothed, linewidth=2)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].grid(True)

        # Plot success rate
        window = 100
        successes = []
        for i in range(len(self.episode_rewards)):
            # Check if agent reached goal (high reward)
            success = self.episode_rewards[i] > 50
            successes.append(success)

        success_rates = []
        for i in range(window, len(successes)):
            rate = np.mean(successes[i-window:i]) * 100
            success_rates.append(rate)

        axes[1, 1].plot(success_rates, linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title(f'Success Rate (last {window} episodes)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'training_metrics.png')
        plt.savefig(filepath, dpi=150)
        print(f"\nTraining metrics saved to {filepath}")
        plt.close()


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train DQN agent on maze')
    parser.add_argument('--maze-size', type=int, default=10,
                       help='Size of the maze (default: 10)')
    parser.add_argument('--render-size', type=int, default=84,
                       help='Size of rendered images (default: 84)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Frequency to save model (default: 100)')
    parser.add_argument('--record-freq', type=int, default=50,
                       help='Frequency to record episodes as .pkl (0=disable, default: 50)')
    parser.add_argument('--live-viz', action='store_true',
                       help='Enable live training visualization')

    args = parser.parse_args()

    # Create trainer
    trainer = Trainer(
        maze_size=args.maze_size,
        render_size=args.render_size,
        num_episodes=args.episodes,
        save_freq=args.save_freq,
        record_freq=args.record_freq,
        enable_live_viz=args.live_viz
    )

    # Train
    trainer.train(render=args.render)

    # Plot metrics
    trainer.plot_training_metrics()


if __name__ == '__main__':
    main()
