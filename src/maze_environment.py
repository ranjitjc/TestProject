"""
Maze Environment for Deep Reinforcement Learning
Provides a visual maze that an agent can navigate using RL
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class MazeEnvironment:
    """
    A customizable maze environment with visual rendering.

    The agent must navigate from start (S) to goal (G) while avoiding walls.
    """

    # Action space
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, maze_size: int = 10, render_size: int = 400):
        """
        Initialize the maze environment.

        Args:
            maze_size: Size of the maze (maze_size x maze_size)
            render_size: Size of the rendered image in pixels
        """
        self.maze_size = maze_size
        self.render_size = render_size
        self.cell_size = render_size // maze_size

        # Define maze layout (0 = path, 1 = wall)
        self.maze = self._generate_maze()

        # Find start and goal positions
        self.start_pos = self._find_start_position()
        self.goal_pos = self._find_goal_position()

        # Current agent position
        self.agent_pos = None

        # Action mapping
        self.actions = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1)
        }

        self.num_actions = len(self.actions)
        self.steps = 0
        self.max_steps = maze_size * maze_size * 2

    def _generate_maze(self) -> np.ndarray:
        """Generate a maze layout using recursive backtracking."""
        maze = np.ones((self.maze_size, self.maze_size), dtype=np.int32)

        def carve_path(x: int, y: int):
            maze[x, y] = 0
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            np.random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size and maze[nx, ny] == 1:
                    maze[x + dx, y + dy] = 0
                    carve_path(nx, ny)

        # Start carving from (1, 1)
        if self.maze_size > 2:
            carve_path(1, 1)
        else:
            maze = np.zeros((self.maze_size, self.maze_size), dtype=np.int32)

        return maze

    def _find_start_position(self) -> Tuple[int, int]:
        """Find a valid start position (top-left area)."""
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (0, 0)

    def _find_goal_position(self) -> Tuple[int, int]:
        """Find a valid goal position (bottom-right area)."""
        for i in range(self.maze_size - 1, -1, -1):
            for j in range(self.maze_size - 1, -1, -1):
                if self.maze[i, j] == 0 and (i, j) != self.start_pos:
                    return (i, j)
        return (self.maze_size - 1, self.maze_size - 1)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation (visual representation of maze)
        """
        self.agent_pos = list(self.start_pos)
        self.steps = 0
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.

        Args:
            action: Action to take (UP, DOWN, LEFT, RIGHT)

        Returns:
            observation: Visual state after action
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        self.steps += 1

        # Calculate new position
        dx, dy = self.actions[action]
        new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]

        # Check if move is valid
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        done = self._is_goal_reached() or self.steps >= self.max_steps

        info = {
            'steps': self.steps,
            'position': tuple(self.agent_pos)
        }

        return self._get_observation(), reward, done, info

    def _is_valid_position(self, pos: list) -> bool:
        """Check if position is within bounds and not a wall."""
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.maze[x, y] == 0
        return False

    def _is_goal_reached(self) -> bool:
        """Check if agent reached the goal."""
        return tuple(self.agent_pos) == self.goal_pos

    def _calculate_reward(self) -> float:
        """Calculate reward based on current state."""
        if self._is_goal_reached():
            return 100.0  # Large positive reward for reaching goal

        # Distance-based reward (encourage moving closer to goal)
        dist_to_goal = np.sqrt(
            (self.agent_pos[0] - self.goal_pos[0]) ** 2 +
            (self.agent_pos[1] - self.goal_pos[1]) ** 2
        )

        # Small negative reward for each step (encourage efficiency)
        return -0.1 - dist_to_goal * 0.01

    def _get_observation(self) -> np.ndarray:
        """
        Get visual observation of the current state.

        Returns:
            RGB image of the maze with agent and goal positions
        """
        # Create RGB image
        img = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)

        # Draw maze
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if self.maze[i, j] == 1:  # Wall
                    color = (50, 50, 50)  # Dark gray
                else:  # Path
                    color = (255, 255, 255)  # White

                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # Draw goal
        gx = self.goal_pos[1] * self.cell_size + self.cell_size // 2
        gy = self.goal_pos[0] * self.cell_size + self.cell_size // 2
        cv2.circle(img, (gx, gy), self.cell_size // 3, (0, 255, 0), -1)  # Green

        # Draw agent
        ax = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        ay = self.agent_pos[0] * self.cell_size + self.cell_size // 2
        cv2.circle(img, (ax, ay), self.cell_size // 3, (255, 0, 0), -1)  # Blue

        # Add grid lines
        for i in range(self.maze_size + 1):
            pos = i * self.cell_size
            cv2.line(img, (pos, 0), (pos, self.render_size), (200, 200, 200), 1)
            cv2.line(img, (0, pos), (self.render_size, pos), (200, 200, 200), 1)

        return img

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode: 'human' for display, 'rgb_array' for returning array
        """
        img = self._get_observation()

        if mode == 'human':
            try:
                cv2.imshow('Maze Environment', img)
                cv2.waitKey(1)
            except cv2.error:
                # Headless mode - cv2.imshow not available
                pass

        return img

    def close(self):
        """Clean up resources."""
        try:
            cv2.destroyAllWindows()
        except (cv2.error, AttributeError):
            # Headless mode - cv2.destroyAllWindows not available
            pass
