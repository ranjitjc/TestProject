"""
Demo script to visualize a trained agent solving the maze
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from src.maze_environment import MazeEnvironment
from src.dqn_agent import DQNAgent


def demo_agent(model_path: str, maze_size: int = 10, render_size: int = 84,
               num_episodes: int = 5, delay: int = 100):
    """
    Demonstrate a trained agent solving mazes.

    Args:
        model_path: Path to trained model
        maze_size: Size of the maze
        render_size: Size of rendered images
        num_episodes: Number of episodes to run
        delay: Delay between steps in milliseconds
    """
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
    print("Press 'q' to quit, 'n' for next episode\n")

    successes = 0
    total_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0

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

            # Show image
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
                # Headless mode - cv2.imshow not available
                # Continue without visualization
                import time
                time.sleep(delay / 1000.0)

            # Select action (no exploration)
            action = agent.select_action(state, training=False)

            # Take action
            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward
            steps += 1

            # Check for timeout
            if steps > maze_size * maze_size * 2:
                print("  Episode timed out!")
                break

        # Episode finished
        total_steps.append(steps)

        if env._is_goal_reached():
            successes += 1
            print(f"  SUCCESS! Reached goal in {steps} steps")
            print(f"  Total reward: {episode_reward:.2f}")
        else:
            print(f"  Failed to reach goal in {steps} steps")
            print(f"  Total reward: {episode_reward:.2f}")

    # Summary statistics
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)
    print(f"Success Rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Average Steps: {np.mean(total_steps):.1f}")
    print(f"Min Steps: {np.min(total_steps)}")
    print(f"Max Steps: {np.max(total_steps)}")

    env.close()
    print("\nDemo completed!")


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

    args = parser.parse_args()

    demo_agent(
        model_path=args.model,
        maze_size=args.maze_size,
        render_size=args.render_size,
        num_episodes=args.episodes,
        delay=args.delay
    )


if __name__ == '__main__':
    main()
