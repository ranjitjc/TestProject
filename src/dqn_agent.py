"""
DQN Agent with Experience Replay and Target Network
Implements the Deep Q-Learning algorithm
"""

import random
import numpy as np
from collections import deque
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.dqn_network import DQNNetwork, DuelingDQNNetwork


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        use_dueling: bool = True,
        device: str = None
    ):
        """
        Initialize DQN Agent.

        Args:
            input_shape: Shape of input (height, width, channels)
            num_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            use_dueling: Whether to use Dueling DQN architecture
            device: Device to use (cuda/cpu)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Convert input shape to (channels, height, width) for PyTorch
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.num_actions = num_actions

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        network_class = DuelingDQNNetwork if use_dueling else DQNNetwork
        self.policy_net = network_class(self.input_shape, num_actions).to(self.device)
        self.target_net = network_class(self.input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training metrics
        self.steps = 0
        self.episodes = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state (image)
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor and move to device."""
        # Convert from (H, W, C) to (C, H, W)
        state = np.transpose(state, (2, 0, 1))
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state_tensor

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step.

        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(np.transpose(states, (0, 3, 1, 2))).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.transpose(next_states, (0, 3, 1, 2))).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save the agent's network."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the agent's network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"Model loaded from {filepath}")
