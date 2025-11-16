#!/usr/bin/env python3
"""
Quick test to check if rendering works
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from src.maze_environment import MazeEnvironment

# Create environment
env = MazeEnvironment(maze_size=5, render_size=84)

# Reset and get first state
state = env.reset()

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Test rendering with save path
print("Testing render with save_path...")
env.render('human', save_path='outputs/test_render.png')

# Check if file was created
if os.path.exists('outputs/test_render.png'):
    print("✅ SUCCESS: outputs/test_render.png was created!")
    file_size = os.path.getsize('outputs/test_render.png')
    print(f"   File size: {file_size} bytes")
else:
    print("❌ FAILED: outputs/test_render.png was NOT created")

# Check headless detection
is_headless = os.environ.get('DISPLAY') is None
print(f"\nHeadless mode detected: {is_headless}")
print(f"DISPLAY env variable: {os.environ.get('DISPLAY', 'Not set')}")
