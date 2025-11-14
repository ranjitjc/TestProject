#!/usr/bin/env python3
"""
Main entry point for Visual Maze Solving with Deep Reinforcement Learning

This script provides a simple interface to train and demo the DQN agent.
"""

import sys
import argparse


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║   Visual Maze Solving with Deep Reinforcement Learning      ║
    ║                    Using PyTorch & DQN                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        description='Visual Maze Solving with Deep Reinforcement Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new agent (quick training)
  python main.py train --episodes 500

  # Train with larger maze
  python main.py train --maze-size 15 --episodes 1000

  # Demo trained agent
  python main.py demo

  # Demo with custom model
  python main.py demo --model ./models/dqn_checkpoint_ep500.pth

For more options, use:
  python main.py train --help
  python main.py demo --help
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new DQN agent')
    train_parser.add_argument('--maze-size', type=int, default=10,
                             help='Size of the maze (default: 10)')
    train_parser.add_argument('--render-size', type=int, default=84,
                             help='Size of rendered images (default: 84)')
    train_parser.add_argument('--episodes', type=int, default=1000,
                             help='Number of training episodes (default: 1000)')
    train_parser.add_argument('--render', action='store_true',
                             help='Render the environment during training')
    train_parser.add_argument('--save-freq', type=int, default=100,
                             help='Frequency to save model (default: 100)')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Demo a trained agent')
    demo_parser.add_argument('--model', type=str, default='./models/dqn_final.pth',
                            help='Path to trained model (default: ./models/dqn_final.pth)')
    demo_parser.add_argument('--maze-size', type=int, default=10,
                            help='Size of the maze (default: 10)')
    demo_parser.add_argument('--render-size', type=int, default=84,
                            help='Size of rendered images - must match training (default: 84)')
    demo_parser.add_argument('--episodes', type=int, default=5,
                            help='Number of episodes to run (default: 5)')
    demo_parser.add_argument('--delay', type=int, default=200,
                            help='Delay between steps in ms (default: 200)')

    args = parser.parse_args()

    if args.command == 'train':
        print("\n🚀 Starting training...\n")
        from src.train import Trainer
        trainer = Trainer(
            maze_size=args.maze_size,
            render_size=args.render_size,
            num_episodes=args.episodes,
            save_freq=args.save_freq
        )
        trainer.train(render=args.render)
        trainer.plot_training_metrics()

    elif args.command == 'demo':
        print("\n🎮 Starting demo...\n")
        from demo import demo_agent
        demo_agent(
            model_path=args.model,
            maze_size=args.maze_size,
            render_size=args.render_size,
            num_episodes=args.episodes,
            delay=args.delay
        )

    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python main.py train --episodes 500")
        print("   python main.py demo")


if __name__ == '__main__':
    main()
