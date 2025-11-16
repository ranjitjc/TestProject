"""
Episode replay system for saving and replaying agent episodes
Allows frame-by-frame analysis and comparison
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2


class EpisodeRecorder:
    """Records episode data for later replay."""

    def __init__(self):
        """Initialize episode recorder."""
        self.reset()

    def reset(self):
        """Start recording a new episode."""
        self.frames = []
        self.actions = []
        self.rewards = []
        self.states = []
        self.metadata = {}

    def record_step(self, state: np.ndarray, action: int,
                   reward: float, info: dict = None):
        """
        Record a single step.

        Args:
            state: Visual observation
            action: Action taken
            reward: Reward received
            info: Additional information
        """
        self.frames.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if info:
            self.states.append(info.get('position', None))

    def finalize(self, success: bool, total_steps: int,
                total_reward: float, episode_num: int = None):
        """
        Finalize episode recording.

        Args:
            success: Whether episode succeeded
            total_steps: Total steps taken
            total_reward: Total reward accumulated
            episode_num: Episode number
        """
        self.metadata = {
            'success': success,
            'total_steps': total_steps,
            'total_reward': total_reward,
            'episode_num': episode_num,
            'num_frames': len(self.frames)
        }

    def save(self, filepath: str):
        """
        Save episode to file.

        Args:
            filepath: Path to save file
        """
        episode_data = {
            'frames': self.frames,
            'actions': self.actions,
            'rewards': self.rewards,
            'states': self.states,
            'metadata': self.metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)

        print(f"Episode saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'EpisodeRecorder':
        """
        Load episode from file.

        Args:
            filepath: Path to episode file

        Returns:
            Loaded episode recorder
        """
        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)

        recorder = EpisodeRecorder()
        recorder.frames = episode_data['frames']
        recorder.actions = episode_data['actions']
        recorder.rewards = episode_data['rewards']
        recorder.states = episode_data.get('states', [])
        recorder.metadata = episode_data['metadata']

        return recorder


class EpisodeReplayer:
    """Replays recorded episodes."""

    ACTION_NAMES = {
        0: 'UP',
        1: 'DOWN',
        2: 'LEFT',
        3: 'RIGHT'
    }

    def __init__(self, episode: EpisodeRecorder):
        """
        Initialize replayer.

        Args:
            episode: Recorded episode to replay
        """
        self.episode = episode
        self.current_frame = 0

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Frame image
        """
        if 0 <= frame_idx < len(self.episode.frames):
            return self.episode.frames[frame_idx]
        return None

    def get_annotated_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get frame with annotations.

        Args:
            frame_idx: Frame index

        Returns:
            Annotated frame
        """
        frame = self.get_frame(frame_idx)
        if frame is None:
            return None

        # Create larger frame for annotations
        h, w = frame.shape[:2]
        annotated = np.ones((h + 70, w, 3), dtype=np.uint8) * 255

        # Place original frame
        annotated[:h, :, :] = frame

        # Add annotations
        action = self.episode.actions[frame_idx] if frame_idx < len(self.episode.actions) else -1
        reward = self.episode.rewards[frame_idx] if frame_idx < len(self.episode.rewards) else 0
        cumulative_reward = sum(self.episode.rewards[:frame_idx+1])

        action_name = self.ACTION_NAMES.get(action, 'N/A')

        # Text annotations with smaller font for small frames
        font_scale = 0.35
        thickness = 1
        line_height = 15

        cv2.putText(annotated, f"F:{frame_idx + 1}/{len(self.episode.frames)}",
                   (5, h + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated, f"A:{action_name}",
                   (5, h + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        cv2.putText(annotated, f"R:{reward:.1f}",
                   (5, h + line_height*3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 128, 0), thickness)
        cv2.putText(annotated, f"Tot:{cumulative_reward:.1f}",
                   (5, h + line_height*4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 0, 128), thickness)

        # Success indicator
        if self.episode.metadata.get('success', False):
            cv2.putText(annotated, "WIN",
                       (w-30, h + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), thickness)

        return annotated

    def play(self, delay: int = 100, headless: bool = False):
        """
        Play episode.

        Args:
            delay: Delay between frames in ms
            headless: If True, don't show window (just iterate)
        """
        print(f"\nReplaying episode:")
        print(f"  Total steps: {self.episode.metadata.get('total_steps', 'N/A')}")
        print(f"  Total reward: {self.episode.metadata.get('total_reward', 'N/A'):.2f}")
        print(f"  Success: {self.episode.metadata.get('success', False)}")
        print("\nControls: 'q' to quit, 'p' to pause, 'n' for next frame, 'b' for previous frame")

        paused = False

        for frame_idx in range(len(self.episode.frames)):
            annotated = self.get_annotated_frame(frame_idx)

            if not headless:
                try:
                    cv2.imshow('Episode Replay', annotated)

                    # Handle key presses
                    if paused:
                        key = cv2.waitKey(0)
                    else:
                        key = cv2.waitKey(delay)

                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif key == ord('n') and frame_idx < len(self.episode.frames) - 1:
                        frame_idx += 1
                    elif key == ord('b') and frame_idx > 0:
                        frame_idx -= 1
                except cv2.error:
                    # Headless environment
                    import time
                    time.sleep(delay / 1000.0)
            else:
                import time
                time.sleep(delay / 1000.0)

        if not headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        print("\nReplay finished")

    def export_video(self, filepath: str, fps: int = 10):
        """
        Export episode as video file.

        Args:
            filepath: Output video file path
            fps: Frames per second
        """
        if len(self.episode.frames) == 0:
            print("No frames to export")
            return

        # Get frame dimensions
        frame = self.get_annotated_frame(0)
        h, w = frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

        # Write frames
        for frame_idx in range(len(self.episode.frames)):
            annotated = self.get_annotated_frame(frame_idx)
            out.write(annotated)

        out.release()
        print(f"Video exported to {filepath}")

    def export_frames_as_images(self, output_dir: str, prefix: str = "frame"):
        """
        Export all frames as individual images.

        Args:
            output_dir: Directory to save images
            prefix: Prefix for image filenames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Exporting {len(self.episode.frames)} frames to {output_dir}...")

        for frame_idx in range(len(self.episode.frames)):
            annotated = self.get_annotated_frame(frame_idx)
            filename = f"{prefix}_{frame_idx:04d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, annotated)

        print(f"  ✓ Exported {len(self.episode.frames)} frames")
        print(f"  First frame: {output_dir}/{prefix}_0000.png")
        print(f"  Last frame: {output_dir}/{prefix}_{len(self.episode.frames)-1:04d}.png")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get episode summary.

        Returns:
            Dictionary with episode summary
        """
        return {
            'num_frames': len(self.episode.frames),
            'total_reward': sum(self.episode.rewards),
            'average_reward': np.mean(self.episode.rewards) if self.episode.rewards else 0,
            'max_reward': max(self.episode.rewards) if self.episode.rewards else 0,
            'min_reward': min(self.episode.rewards) if self.episode.rewards else 0,
            'metadata': self.episode.metadata
        }
