"""
Test script to verify the implementation is working correctly
"""

import sys
import numpy as np
import torch

print("=" * 60)
print("Testing Visual Maze Solving with Deep Reinforcement Learning")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from src.maze_environment import MazeEnvironment
    from src.dqn_network import DQNNetwork, DuelingDQNNetwork
    from src.dqn_agent import DQNAgent, ReplayBuffer
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test PyTorch
print("\n2. Testing PyTorch...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ PyTorch error: {e}")
    sys.exit(1)

# Test maze environment
print("\n3. Testing Maze Environment...")
try:
    env = MazeEnvironment(maze_size=5, render_size=84)
    state = env.reset()
    print(f"   ✓ Environment created successfully")
    print(f"   ✓ State shape: {state.shape}")
    print(f"   ✓ Number of actions: {env.num_actions}")
    print(f"   ✓ Start position: {env.start_pos}")
    print(f"   ✓ Goal position: {env.goal_pos}")

    # Test step
    next_state, reward, done, info = env.step(0)
    print(f"   ✓ Step executed successfully")
    print(f"   ✓ Reward: {reward:.2f}")
    env.close()
except Exception as e:
    print(f"   ✗ Environment error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test neural network
print("\n4. Testing Neural Networks...")
try:
    input_shape = (3, 84, 84)
    num_actions = 4

    # Test standard DQN
    dqn = DQNNetwork(input_shape, num_actions)
    test_input = torch.randn(1, 3, 84, 84)
    output = dqn(test_input)
    print(f"   ✓ DQN Network created")
    print(f"   ✓ Output shape: {output.shape}")

    # Test Dueling DQN
    dueling_dqn = DuelingDQNNetwork(input_shape, num_actions)
    output = dueling_dqn(test_input)
    print(f"   ✓ Dueling DQN Network created")
    print(f"   ✓ Output shape: {output.shape}")
except Exception as e:
    print(f"   ✗ Neural network error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test replay buffer
print("\n5. Testing Replay Buffer...")
try:
    buffer = ReplayBuffer(capacity=100)

    # Add some transitions
    for i in range(10):
        state = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        done = False
        buffer.push(state, action, reward, next_state, done)

    print(f"   ✓ Replay buffer created")
    print(f"   ✓ Buffer size: {len(buffer)}")

    # Test sampling
    if len(buffer) >= 5:
        batch = buffer.sample(5)
        print(f"   ✓ Sampling successful")
except Exception as e:
    print(f"   ✗ Replay buffer error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test DQN agent
print("\n6. Testing DQN Agent...")
try:
    env = MazeEnvironment(maze_size=5, render_size=84)
    input_shape = (84, 84, 3)
    agent = DQNAgent(
        input_shape=input_shape,
        num_actions=env.num_actions,
        use_dueling=True
    )

    print(f"   ✓ Agent created successfully")
    print(f"   ✓ Device: {agent.device}")

    # Test action selection
    state = env.reset()
    action = agent.select_action(state, training=False)
    print(f"   ✓ Action selection works")
    print(f"   ✓ Selected action: {action}")

    # Test storing transition
    next_state, reward, done, info = env.step(action)
    agent.store_transition(state, action, reward, next_state, done)
    print(f"   ✓ Transition stored")

    env.close()
except Exception as e:
    print(f"   ✗ Agent error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test short training loop
print("\n7. Testing Short Training Loop...")
try:
    env = MazeEnvironment(maze_size=5, render_size=84)
    agent = DQNAgent(
        input_shape=(84, 84, 3),
        num_actions=env.num_actions,
        use_dueling=True
    )

    # Run a few episodes
    for episode in range(3):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done and steps < 50:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # Train if buffer has enough samples
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()

            state = next_state
            episode_reward += reward
            steps += 1

        agent.update_epsilon()
        print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}")

    print(f"   ✓ Training loop works correctly")
    env.close()
except Exception as e:
    print(f"   ✗ Training loop error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
print("\nYou can now:")
print("  1. Train an agent: python main.py train --episodes 500")
print("  2. Demo an agent: python main.py demo")
print("\nFor more information, see README.md")
