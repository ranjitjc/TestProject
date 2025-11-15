"""
Training script for DQN agent on maze environment
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from src.maze_environment import MazeEnvironment
from src.dqn_agent import DQNAgent


class Trainer:
    """Trainer class for DQN agent on maze environment."""

    def __init__(
        self,
        maze_size: int = 10,
        render_size: int = 84,
        num_episodes: int = 1000,
        save_freq: int = 100,
        model_dir: str = './models',
        output_dir: str = './outputs'
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
        """
        self.maze_size = maze_size
        self.render_size = render_size
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        self.model_dir = model_dir
        self.output_dir = output_dir

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

    def train(self, render: bool = False):
        """
        Train the agent.

        Args:
            render: Whether to render the environment during training
        """
        print(f"Training DQN agent on {self.maze_size}x{self.maze_size} maze")
        print(f"Device: {self.agent.device}")
        print(f"Number of episodes: {self.num_episodes}")
        print("-" * 50)

        recent_successes = []

        for episode in tqdm(range(self.num_episodes), desc="Training"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            done = False

            while not done:
                # Select and perform action
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train agent
                loss = self.agent.train_step()
                episode_loss += loss

                # Update state
                state = next_state
                episode_reward += reward
                steps += 1

                # Render if requested
                if render and episode % 10 == 0:
                    self.env.render('human')
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

        # Close environment
        self.env.close()

        print("\nTraining completed!")

    def save_model(self, filename: str):
        """Save the model."""
        filepath = os.path.join(self.model_dir, filename)
        self.agent.save(filepath)

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

    args = parser.parse_args()

    # Create trainer
    trainer = Trainer(
        maze_size=args.maze_size,
        render_size=args.render_size,
        num_episodes=args.episodes,
        save_freq=args.save_freq
    )

    # Train
    trainer.train(render=args.render)

    # Plot metrics
    trainer.plot_training_metrics()


if __name__ == '__main__':
    main()
