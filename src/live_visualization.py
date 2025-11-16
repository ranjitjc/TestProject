"""
Real-time visualization for training progress
Displays live plots of metrics during training
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import threading
import time


class LiveTrainingVisualizer:
    """
    Real-time visualization of training metrics.

    Updates plots dynamically as training progresses.
    """

    def __init__(self, window_size: int = 100, update_interval: int = 10, output_dir: str = 'outputs'):
        """
        Initialize live visualizer.

        Args:
            window_size: Number of recent episodes to display
            update_interval: Update plots every N episodes
            output_dir: Directory to save visualization images
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.output_dir = output_dir

        # Metric storage
        self.episodes = []
        self.rewards = []
        self.lengths = []
        self.losses = []
        self.epsilons = []
        self.success_rates = []

        # Figure setup
        self.fig = None
        self.axes = None
        self.lines = {}

        # State
        self.is_initialized = False
        self.episode_count = 0

    def initialize(self):
        """Initialize the plotting figure."""
        if self.is_initialized:
            return

        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('Live Training Visualization', fontsize=16)

        # Configure subplots
        titles = [
            'Episode Rewards',
            'Episode Length',
            'Training Loss',
            'Epsilon (Exploration)',
            'Success Rate',
            'Cumulative Statistics'
        ]

        ylabels = [
            'Reward',
            'Steps',
            'Loss',
            'Epsilon',
            'Success Rate (%)',
            'Value'
        ]

        for idx, (ax, title, ylabel) in enumerate(zip(self.axes.flat, titles, ylabels)):
            ax.set_title(title)
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.is_initialized = True

    def update(self, episode: int, reward: float, length: int,
               loss: float, epsilon: float, success: bool):
        """
        Update metrics with new episode data.

        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length (steps)
            loss: Average training loss
            epsilon: Current exploration rate
            success: Whether episode succeeded
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.lengths.append(length)
        self.losses.append(loss)
        self.epsilons.append(epsilon)

        # Calculate rolling success rate
        # Build list of recent successes (current + previous episodes)
        current_success = 1 if success else 0

        # Get previous successes from success_rates (convert back from percentage)
        if len(self.success_rates) > 0:
            # Estimate number of previous successes from last success rate
            prev_count = min(len(self.success_rates), 99)
            prev_success_rate = self.success_rates[-1] / 100.0
            prev_successes = int(prev_success_rate * prev_count)
            total_successes = current_success + prev_successes
            total_episodes = prev_count + 1
            success_rate = (total_successes / total_episodes) * 100
        else:
            # First episode
            success_rate = current_success * 100.0

        self.success_rates.append(success_rate)

        self.episode_count += 1

        # Update plot if interval reached
        if self.episode_count % self.update_interval == 0:
            self.render()

    def render(self):
        """Render the current state of all plots."""
        if not self.is_initialized:
            self.initialize()

        # Get windowed data
        start_idx = max(0, len(self.episodes) - self.window_size)

        episodes = self.episodes[start_idx:]
        rewards = self.rewards[start_idx:]
        lengths = self.lengths[start_idx:]
        losses = self.losses[start_idx:]
        epsilons = self.epsilons[start_idx:]
        success_rates = self.success_rates[start_idx:]

        # Clear and update each subplot
        axes = self.axes.flat

        # Plot 1: Rewards
        axes[0].clear()
        axes[0].plot(episodes, rewards, 'b-', alpha=0.3, label='Raw')
        if len(rewards) > 10:
            smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
            axes[0].plot(episodes[9:], smoothed, 'b-', linewidth=2, label='Smoothed')
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Episode Length
        axes[1].clear()
        axes[1].plot(episodes, lengths, 'g-', alpha=0.3, label='Raw')
        if len(lengths) > 10:
            smoothed = np.convolve(lengths, np.ones(10)/10, mode='valid')
            axes[1].plot(episodes[9:], smoothed, 'g-', linewidth=2, label='Smoothed')
        axes[1].set_title('Episode Length')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Loss
        axes[2].clear()
        axes[2].plot(episodes, losses, 'r-', alpha=0.5)
        axes[2].set_title('Training Loss')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Epsilon
        axes[3].clear()
        axes[3].plot(episodes, epsilons, 'orange', linewidth=2)
        axes[3].set_title('Epsilon (Exploration Rate)')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Epsilon')
        axes[3].grid(True, alpha=0.3)

        # Plot 5: Success Rate
        axes[4].clear()
        axes[4].plot(episodes, success_rates, 'purple', linewidth=2)
        axes[4].fill_between(episodes, 0, success_rates, alpha=0.3, color='purple')
        axes[4].set_title('Success Rate (Last 100 Episodes)')
        axes[4].set_xlabel('Episode')
        axes[4].set_ylabel('Success Rate (%)')
        axes[4].set_ylim([0, 100])
        axes[4].grid(True, alpha=0.3)

        # Plot 6: Cumulative Statistics
        axes[5].clear()
        if len(rewards) > 0:
            avg_reward = np.mean(rewards)
            avg_length = np.mean(lengths)
            current_success = success_rates[-1] if success_rates else 0

            stats_text = f"Average Reward: {avg_reward:.2f}\n"
            stats_text += f"Average Length: {avg_length:.1f}\n"
            stats_text += f"Success Rate: {current_success:.1f}%\n"
            stats_text += f"Current Epsilon: {epsilons[-1]:.3f}\n"
            stats_text += f"Episodes: {episodes[-1]}"

            axes[5].text(0.1, 0.5, stats_text, fontsize=12,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[5].axis('off')

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Auto-save in headless environments
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        self.fig.savefig(f'{self.output_dir}/live_training_viz.png', dpi=150, bbox_inches='tight')
        print(f"📊 Visualization updated → {self.output_dir}/live_training_viz.png")

    def save(self, filepath: str):
        """Save the current visualization."""
        if self.fig:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Live visualization saved to {filepath}")

    def close(self):
        """Close the visualization."""
        if self.fig:
            plt.close(self.fig)
            self.is_initialized = False
