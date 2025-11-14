"""
Deep Q-Network (DQN) implementation using PyTorch
Uses a CNN to process visual input from the maze
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Convolutional Neural Network for Deep Q-Learning.

    Processes visual input and outputs Q-values for each action.
    """

    def __init__(self, input_shape: tuple, num_actions: int):
        """
        Initialize the DQN network.

        Args:
            input_shape: Shape of input images (channels, height, width)
            num_actions: Number of possible actions
        """
        super(DQNNetwork, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size after convolutions
        self.feature_size = self._get_conv_output_size(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output_size(self, shape: tuple) -> int:
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self._forward_conv(dummy_input)
            return int(dummy_output.numel())

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Q-values for each action
        """
        # Normalize input to [0, 1]
        x = x.float() / 255.0

        # Convolutional layers
        x = self._forward_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    Often performs better than standard DQN.
    """

    def __init__(self, input_shape: tuple, num_actions: int):
        """
        Initialize the Dueling DQN network.

        Args:
            input_shape: Shape of input images (channels, height, width)
            num_actions: Number of possible actions
        """
        super(DuelingDQNNetwork, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size after convolutions
        self.feature_size = self._get_conv_output_size(input_shape)

        # Value stream
        self.value_fc1 = nn.Linear(self.feature_size, 512)
        self.value_fc2 = nn.Linear(512, 1)

        # Advantage stream
        self.advantage_fc1 = nn.Linear(self.feature_size, 512)
        self.advantage_fc2 = nn.Linear(512, num_actions)

    def _get_conv_output_size(self, shape: tuple) -> int:
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self._forward_conv(dummy_input)
            return int(dummy_output.numel())

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Q-values for each action
        """
        # Normalize input to [0, 1]
        x = x.float() / 255.0

        # Shared convolutional layers
        x = self._forward_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
