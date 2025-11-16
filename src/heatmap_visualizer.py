"""
Heatmap visualization for agent exploration
Shows which positions the agent visits most frequently
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class HeatmapVisualizer:
    """
    Visualizes agent exploration using heatmaps.

    Tracks and displays which maze positions are visited most frequently.
    """

    def __init__(self, maze_size: int):
        """
        Initialize heatmap visualizer.

        Args:
            maze_size: Size of the maze (maze_size x maze_size)
        """
        self.maze_size = maze_size
        self.visit_counts = np.zeros((maze_size, maze_size), dtype=np.int32)
        self.episode_paths = []

    def record_position(self, position: tuple):
        """
        Record a visit to a position.

        Args:
            position: (row, col) position in the maze
        """
        row, col = position
        if 0 <= row < self.maze_size and 0 <= col < self.maze_size:
            self.visit_counts[row, col] += 1

    def record_path(self, path: list):
        """
        Record a complete episode path.

        Args:
            path: List of (row, col) positions visited in episode
        """
        self.episode_paths.append(path)
        for position in path:
            self.record_position(position)

    def create_heatmap(self, maze: np.ndarray = None,
                      start_pos: tuple = None,
                      goal_pos: tuple = None) -> plt.Figure:
        """
        Create a heatmap visualization.

        Args:
            maze: Optional maze layout (0=path, 1=wall)
            start_pos: Optional start position to mark
            goal_pos: Optional goal position to mark

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create heatmap
        heatmap = np.copy(self.visit_counts).astype(float)

        # Mask walls if maze provided
        if maze is not None:
            heatmap = np.ma.masked_where(maze == 1, heatmap)

        # Plot heatmap
        im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visit Count', rotation=270, labelpad=20)

        # Mark walls
        if maze is not None:
            wall_positions = np.argwhere(maze == 1)
            for row, col in wall_positions:
                ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                                          fill=True, color='gray', alpha=0.5))

        # Mark start position
        if start_pos is not None:
            ax.plot(start_pos[1], start_pos[0], 'bs',
                   markersize=15, label='Start', markeredgecolor='white', markeredgewidth=2)

        # Mark goal position
        if goal_pos is not None:
            ax.plot(goal_pos[1], goal_pos[0], 'g*',
                   markersize=20, label='Goal', markeredgecolor='white', markeredgewidth=2)

        # Configure plot
        ax.set_title('Agent Exploration Heatmap\n(Brighter = More Visits)', fontsize=14)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend(loc='upper right')

        # Add grid
        ax.set_xticks(np.arange(self.maze_size))
        ax.set_yticks(np.arange(self.maze_size))
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

        plt.tight_layout()
        return fig

    def create_path_visualization(self, episode_idx: int,
                                  maze: np.ndarray = None,
                                  start_pos: tuple = None,
                                  goal_pos: tuple = None) -> plt.Figure:
        """
        Visualize a specific episode path.

        Args:
            episode_idx: Index of episode to visualize
            maze: Optional maze layout
            start_pos: Start position
            goal_pos: Goal position

        Returns:
            Matplotlib figure
        """
        if episode_idx >= len(self.episode_paths):
            raise ValueError(f"Episode {episode_idx} not found")

        path = self.episode_paths[episode_idx]

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw maze
        if maze is not None:
            maze_display = np.ones_like(maze, dtype=float)
            maze_display[maze == 1] = 0.3  # Walls darker
            ax.imshow(maze_display, cmap='gray', alpha=0.5)

        # Draw path
        if len(path) > 0:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0],
                   'b-', linewidth=2, alpha=0.6, label='Path')

            # Mark positions with dots
            ax.scatter(path_array[:, 1], path_array[:, 0],
                      c=range(len(path)), cmap='viridis',
                      s=50, alpha=0.7, edgecolors='white', linewidths=1)

        # Mark start
        if start_pos is not None:
            ax.plot(start_pos[1], start_pos[0], 'bs',
                   markersize=15, label='Start', markeredgecolor='white', markeredgewidth=2)

        # Mark goal
        if goal_pos is not None:
            ax.plot(goal_pos[1], goal_pos[0], 'g*',
                   markersize=20, label='Goal', markeredgecolor='white', markeredgewidth=2)

        # Configure
        ax.set_title(f'Episode {episode_idx} Path ({len(path)} steps)', fontsize=14)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.5, self.maze_size - 0.5])
        ax.set_ylim([self.maze_size - 0.5, -0.5])

        plt.tight_layout()
        return fig

    def get_statistics(self) -> dict:
        """
        Get exploration statistics.

        Returns:
            Dictionary with statistics
        """
        total_visits = np.sum(self.visit_counts)
        visited_cells = np.count_nonzero(self.visit_counts)
        max_visits = np.max(self.visit_counts)

        return {
            'total_visits': int(total_visits),
            'unique_cells_visited': int(visited_cells),
            'max_visits_single_cell': int(max_visits),
            'total_episodes': len(self.episode_paths),
            'coverage': float(visited_cells) / (self.maze_size ** 2) * 100
        }

    def reset(self):
        """Reset all tracking data."""
        self.visit_counts = np.zeros((self.maze_size, self.maze_size), dtype=np.int32)
        self.episode_paths = []

    def save_heatmap(self, filepath: str, **kwargs):
        """Save heatmap to file."""
        fig = self.create_heatmap(**kwargs)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Heatmap saved to {filepath}")
