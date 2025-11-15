#!/usr/bin/env python3
"""
Quick setup verification - checks if Python syntax is valid
"""

import sys
import py_compile

print("Verifying Python syntax for all source files...")

files = [
    'src/maze_environment.py',
    'src/dqn_network.py',
    'src/dqn_agent.py',
    'src/train.py',
    'main.py',
    'demo.py',
    'test_setup.py'
]

all_valid = True
for filepath in files:
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"  ✓ {filepath}")
    except py_compile.PyCompileError as e:
        print(f"  ✗ {filepath}: {e}")
        all_valid = False

if all_valid:
    print("\n✓ All files have valid Python syntax!")
    print("\nTo install dependencies and run:")
    print("  pip install -r requirements.txt")
    print("  python test_setup.py")
    sys.exit(0)
else:
    print("\n✗ Some files have syntax errors")
    sys.exit(1)
