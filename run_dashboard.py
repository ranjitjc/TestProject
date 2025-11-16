#!/usr/bin/env python3
"""
Launch the web dashboard for training visualization

Usage:
    python run_dashboard.py

Then open your browser to http://localhost:8501
"""

import sys
import subprocess


def main():
    """Run the Streamlit dashboard."""
    print("=" * 60)
    print("  Starting DRL Maze Solver Dashboard")
    print("=" * 60)
    print("\nThe dashboard will open in your browser")
    print("URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/web_dashboard.py",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped")


if __name__ == '__main__':
    main()
