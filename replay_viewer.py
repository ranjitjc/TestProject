#!/usr/bin/env python3
"""
Episode replay viewer
Load and view recorded episodes
"""

import argparse
from pathlib import Path
from src.episode_replay import EpisodeRecorder, EpisodeReplayer


def main():
    """Main replay viewer."""
    parser = argparse.ArgumentParser(description='View recorded episodes')
    parser.add_argument('episode_file', type=str,
                       help='Path to episode file (.pkl)')
    parser.add_argument('--delay', type=int, default=100,
                       help='Delay between frames in ms (default: 100)')
    parser.add_argument('--export-video', type=str, default=None,
                       help='Export episode as video file')
    parser.add_argument('--export-images', type=str, default=None,
                       help='Export episode frames as images to directory')
    parser.add_argument('--fps', type=int, default=10,
                       help='Video FPS (default: 10)')
    parser.add_argument('--summary', action='store_true',
                       help='Show episode summary only')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no display)')

    args = parser.parse_args()

    # Check file exists
    if not Path(args.episode_file).exists():
        print(f"Error: Episode file not found: {args.episode_file}")
        return 1

    # Load episode
    print(f"Loading episode from {args.episode_file}...")
    episode = EpisodeRecorder.load(args.episode_file)
    replayer = EpisodeReplayer(episode)

    # Show summary
    if args.summary:
        summary = replayer.get_summary()
        print("\n" + "=" * 60)
        print("  EPISODE SUMMARY")
        print("=" * 60)
        print(f"Total Frames: {summary['num_frames']}")
        print(f"Total Reward: {summary['total_reward']:.2f}")
        print(f"Average Reward: {summary['average_reward']:.2f}")
        print(f"Max Reward: {summary['max_reward']:.2f}")
        print(f"Min Reward: {summary['min_reward']:.2f}")
        print("\nMetadata:")
        for key, value in summary['metadata'].items():
            print(f"  {key}: {value}")
        return 0

    # Export video
    if args.export_video:
        print(f"Exporting video to {args.export_video}...")
        replayer.export_video(args.export_video, fps=args.fps)
        print("Done!")
        return 0

    # Export images
    if args.export_images:
        print(f"Exporting frames to {args.export_images}...")
        replayer.export_frames_as_images(args.export_images)
        print("Done!")
        return 0

    # Detect headless environment
    import os
    is_headless = (
        args.headless or
        os.environ.get('DISPLAY') is None or
        os.environ.get('TERM') == 'dumb'
    )

    if is_headless:
        print("\n⚠️  Headless environment detected!")
        print("Visual playback is not available in this environment.")
        print("\nAvailable options:")
        print("  1. View summary:")
        print(f"     python replay_viewer.py {args.episode_file} --summary")
        print("  2. Export as video:")
        print(f"     python replay_viewer.py {args.episode_file} --export-video episode.mp4")
        print("  3. Export as images:")
        print(f"     python replay_viewer.py {args.episode_file} --export-images frames/")
        print("\nShowing summary instead...\n")

        # Show summary in headless mode
        summary = replayer.get_summary()
        print("=" * 60)
        print("  EPISODE SUMMARY")
        print("=" * 60)
        print(f"Total Frames: {summary['num_frames']}")
        print(f"Total Reward: {summary['total_reward']:.2f}")
        print(f"Average Reward: {summary['average_reward']:.2f}")
        print(f"Max Reward: {summary['max_reward']:.2f}")
        print(f"Min Reward: {summary['min_reward']:.2f}")
        print("\nMetadata:")
        for key, value in summary['metadata'].items():
            print(f"  {key}: {value}")
        return 0

    # Play episode (only in non-headless environments)
    replayer.play(delay=args.delay)

    return 0


if __name__ == '__main__':
    exit(main())
