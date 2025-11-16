"""
Demo script to visualize a trained agent solving the maze
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless
import matplotlib.pyplot as plt

from src.maze_environment import MazeEnvironment
from src.dqn_agent import DQNAgent
from src.episode_replay import EpisodeRecorder


def demo_agent(model_path: str, maze_size: int = 10, render_size: int = 84,
               num_episodes: int = 5, delay: int = 100, save_dir: str = 'demo_outputs',
               record_episodes: bool = False):
    """
    Demonstrate a trained agent solving mazes.

    Args:
        model_path: Path to trained model
        maze_size: Size of the maze
        render_size: Size of rendered images
        num_episodes: Number of episodes to run
        delay: Delay between steps in milliseconds
        save_dir: Directory to save demo outputs
        record_episodes: Whether to record episodes
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Detect headless mode
    is_headless = os.environ.get('DISPLAY') is None
    if is_headless:
        print("Headless mode detected - saving visualizations to files")
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using: python src/train.py")
        return

    # Initialize environment
    env = MazeEnvironment(maze_size=maze_size, render_size=render_size)

    # Initialize agent
    input_shape = (render_size, render_size, 3)
    agent = DQNAgent(
        input_shape=input_shape,
        num_actions=env.num_actions,
        epsilon_start=0.0,  # No exploration for demo
        epsilon_end=0.0,
        use_dueling=True
    )

    # Load trained model
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"\nDemonstrating trained agent on {maze_size}x{maze_size} maze")
    print(f"Running {num_episodes} episodes...")
    if not is_headless:
        print("Press 'q' to quit, 'n' for next episode")
    print(f"Outputs will be saved to: {save_dir}\n")

    successes = 0
    total_steps = []
    total_rewards = []
    episode_data = []  # Store data for visualization

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0

        # Initialize episode recorder if requested
        recorder = EpisodeRecorder() if record_episodes else None

        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        while not done:
            # Render environment
            img = env.render('rgb_array')

            # Add episode info to image
            info_img = np.ones((80, render_size, 3), dtype=np.uint8) * 255
            cv2.putText(info_img, f"Episode: {episode + 1}/{num_episodes}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(info_img, f"Steps: {steps}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(info_img, f"Reward: {episode_reward:.1f}",
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Combine images
            display_img = np.vstack([img, info_img])

            # Show/save image
            if is_headless:
                # Save current frame in headless mode
                cv2.imwrite(f'{save_dir}/demo_current.png', display_img)
            else:
                try:
                    cv2.imshow('Trained Agent Demo', display_img)
                    key = cv2.waitKey(delay)

                    # Handle key presses
                    if key == ord('q'):
                        print("\nQuitting demo...")
                        env.close()
                        return
                    elif key == ord('n'):
                        print("Skipping to next episode...")
                        break
                except (cv2.error, AttributeError):
                    # Fallback to file saving
                    cv2.imwrite(f'{save_dir}/demo_current.png', display_img)
                    time.sleep(delay / 1000.0)

            # Select action (no exploration)
            action = agent.select_action(state, training=False)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Record step if recording
            if recorder:
                recorder.record_step(state, action, reward, info)

            state = next_state
            episode_reward += reward
            steps += 1

            # Check for timeout
            if steps > maze_size * maze_size * 2:
                print("  Episode timed out!")
                break

        # Episode finished
        total_steps.append(steps)
        total_rewards.append(episode_reward)
        success = env._is_goal_reached()

        # Save recorded episode
        if recorder:
            recorder.episode.metadata['success'] = success
            recorder.episode.metadata['steps'] = steps
            recorder.episode.metadata['total_reward'] = episode_reward
            recorder.save(f'{save_dir}/demo_episode_{episode}.pkl')
            print(f"  Recorded episode saved to {save_dir}/demo_episode_{episode}.pkl")

        if env._is_goal_reached():
            successes += 1
            print(f"  ✅ SUCCESS! Reached goal in {steps} steps")
            print(f"  Total reward: {episode_reward:.2f}")
        else:
            print(f"  ❌ Failed to reach goal in {steps} steps")
            print(f"  Total reward: {episode_reward:.2f}")

        # Store episode data for visualization
        episode_data.append({
            'episode': episode + 1,
            'steps': steps,
            'reward': episode_reward,
            'success': success
        })

    # Summary statistics
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)
    print(f"Success Rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Average Steps: {np.mean(total_steps):.1f}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Min Steps: {np.min(total_steps)}")
    print(f"Max Steps: {np.max(total_steps)}")

    # Generate summary visualization
    print(f"\nGenerating summary visualization...")
    create_demo_summary_viz(episode_data, save_dir)
    print(f"Summary saved to: {save_dir}/demo_summary.png")

    env.close()
    print("\nDemo completed!")


def create_demo_summary_viz(episode_data, save_dir):
    """Create summary visualization of demo performance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Demo Performance Summary', fontsize=16, fontweight='bold')

    episodes = [d['episode'] for d in episode_data]
    steps = [d['steps'] for d in episode_data]
    rewards = [d['reward'] for d in episode_data]
    successes = [d['success'] for d in episode_data]

    # Plot 1: Steps per episode
    axes[0, 0].bar(episodes, steps, color=['green' if s else 'red' for s in successes])
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Steps')
    axes[0, 0].set_title('Steps to Complete Each Episode')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(steps), color='blue', linestyle='--',
                       label=f'Average: {np.mean(steps):.1f}')
    axes[0, 0].legend()

    # Plot 2: Rewards per episode
    axes[0, 1].plot(episodes, rewards, marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].set_title('Reward per Episode')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=np.mean(rewards), color='red', linestyle='--',
                       label=f'Average: {np.mean(rewards):.2f}')
    axes[0, 1].legend()

    # Plot 3: Success/Failure
    success_count = sum(successes)
    failure_count = len(successes) - success_count
    axes[1, 0].pie([success_count, failure_count],
                   labels=['Success', 'Failure'],
                   colors=['green', 'red'],
                   autopct='%1.1f%%',
                   startangle=90)
    axes[1, 0].set_title(f'Success Rate: {success_count}/{len(successes)}')

    # Plot 4: Statistics summary
    stats_text = f"""
    Performance Statistics
    {'='*30}

    Episodes Run: {len(episode_data)}
    Success Rate: {success_count/len(successes)*100:.1f}%

    Steps:
      Average: {np.mean(steps):.1f}
      Min: {np.min(steps)}
      Max: {np.max(steps)}

    Rewards:
      Average: {np.mean(rewards):.2f}
      Min: {np.min(rewards):.2f}
      Max: {np.max(rewards):.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/demo_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Demo trained DQN agent on maze')
    parser.add_argument('--model', type=str, default='./models/dqn_final.pth',
                       help='Path to trained model (default: ./models/dqn_final.pth)')
    parser.add_argument('--maze-size', type=int, default=10,
                       help='Size of the maze (default: 10)')
    parser.add_argument('--render-size', type=int, default=84,
                       help='Size of rendered images - must match training size (default: 84)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run (default: 5)')
    parser.add_argument('--delay', type=int, default=200,
                       help='Delay between steps in ms (default: 200)')
    parser.add_argument('--save-dir', type=str, default='demo_outputs',
                       help='Directory to save demo outputs (default: demo_outputs)')
    parser.add_argument('--record', action='store_true',
                       help='Record episodes for later replay')

    args = parser.parse_args()

    demo_agent(
        model_path=args.model,
        maze_size=args.maze_size,
        render_size=args.render_size,
        num_episodes=args.episodes,
        delay=args.delay,
        save_dir=args.save_dir,
        record_episodes=args.record
    )


if __name__ == '__main__':
    main()
