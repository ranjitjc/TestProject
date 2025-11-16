"""
Streamlit web dashboard for interactive visualization and control
Real-time monitoring of training progress
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
import time


class TrainingDashboard:
    """
    Web-based dashboard for training visualization.

    Provides real-time metrics, interactive controls, and model management.
    """

    def __init__(self, log_dir: str = './outputs', model_dir: str = './models', demo_dir: str = './demo_outputs'):
        """
        Initialize dashboard.

        Args:
            log_dir: Directory for training logs
            model_dir: Directory for saved models
            demo_dir: Directory for demo outputs
        """
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.demo_dir = Path(demo_dir)

        # Create directories if needed
        self.log_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.demo_dir.mkdir(exist_ok=True)

    def load_training_data(self, log_file: str = 'training_log.json') -> pd.DataFrame:
        """
        Load training data from log file.

        Args:
            log_file: Name of log file

        Returns:
            DataFrame with training data
        """
        log_path = self.log_dir / log_file

        if not log_path.exists():
            return pd.DataFrame()

        with open(log_path, 'r') as f:
            data = json.load(f)

        return pd.DataFrame(data)

    def create_metrics_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Create interactive metrics plot.

        Args:
            df: Training data

        Returns:
            Plotly figure
        """
        if df.empty:
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Episode Length',
                          'Training Loss', 'Success Rate'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Rewards
        fig.add_trace(
            go.Scatter(x=df['episode'], y=df['reward'],
                      mode='lines', name='Reward',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )

        # Episode length
        fig.add_trace(
            go.Scatter(x=df['episode'], y=df['length'],
                      mode='lines', name='Length',
                      line=dict(color='green', width=1)),
            row=1, col=2
        )

        # Loss
        if 'loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['episode'], y=df['loss'],
                          mode='lines', name='Loss',
                          line=dict(color='red', width=1)),
                row=2, col=1
            )

        # Success rate
        if 'success_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['episode'], y=df['success_rate'],
                          mode='lines', name='Success Rate',
                          line=dict(color='purple', width=2),
                          fill='tozeroy'),
                row=2, col=2
            )

        # Update layout
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=2)

        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Steps", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=2, col=2)

        fig.update_layout(height=700, showlegend=False,
                         title_text="Training Metrics Dashboard")

        return fig

    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="DRL Maze Solver Dashboard",
            page_icon="🧩",
            layout="wide"
        )

        # Sidebar
        with st.sidebar:
            st.header("📊 Navigation")

            # Page selector
            page = st.radio(
                "Select Page",
                ["Training Dashboard", "Demo Dashboard", "Episode Viewer"],
                key="page_selector"
            )

            st.markdown("---")

            # Page-specific sidebar content
            if page == "Training Dashboard":
                st.header("⚙️ Settings")
                refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 5)
                auto_refresh = st.checkbox("Auto-refresh", value=True)

                st.markdown("---")

                # Model selection
                st.subheader("📦 Models")
                models = list(self.model_dir.glob("*.pth"))
                if models:
                    selected_model = st.selectbox(
                        "Select Model",
                        [m.name for m in models]
                    )
                else:
                    st.info("No models found")

                st.markdown("---")

                # Manual refresh button
                if st.button("🔄 Refresh Now"):
                    st.rerun()

            elif page == "Demo Dashboard":
                refresh_rate = 5
                auto_refresh = False

                # Demo-specific controls
                st.subheader("🎮 Demo Info")
                demo_episodes = list(self.demo_dir.glob('demo_episode_*.pkl'))
                if demo_episodes:
                    st.metric("Recorded Episodes", len(demo_episodes))
                else:
                    st.info("No recorded episodes")

                st.markdown("---")

                # Quick actions
                st.subheader("⚡ Quick Actions")
                st.code("python demo.py --episodes 10 --record", language="bash")

                st.markdown("---")

                # Manual refresh button
                if st.button("🔄 Refresh Now"):
                    st.rerun()

            elif page == "Episode Viewer":
                refresh_rate = 5
                auto_refresh = False

                # Episode viewer info
                st.subheader("📁 Viewer Files")
                viewer_path = Path("viewer.html")
                if viewer_path.exists():
                    st.success("✓ Viewer available")
                else:
                    st.warning("No viewer.html found")

                st.markdown("---")

                # Manual refresh button
                if st.button("🔄 Refresh Now"):
                    st.rerun()

        # Route to appropriate page
        if page == "Training Dashboard":
            self.render_training_dashboard()
        elif page == "Demo Dashboard":
            self.render_demo_dashboard()
        else:
            self.render_episode_viewer()

        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()

    def render_training_dashboard(self):
        """Render the training dashboard page."""
        st.title("🧩 Visual Maze Solving - Training Dashboard")
        st.markdown("---")

        # Main content
        col1, col2, col3 = st.columns(3)

        # Load training data
        df = self.load_training_data()

        if not df.empty:
            # Display key metrics
            latest = df.iloc[-1]

            with col1:
                st.metric(
                    label="Latest Reward",
                    value=f"{latest['reward']:.2f}",
                    delta=f"{latest['reward'] - df.iloc[-2]['reward']:.2f}" if len(df) > 1 else None
                )

            with col2:
                st.metric(
                    label="Latest Episode Length",
                    value=f"{int(latest['length'])} steps",
                    delta=f"{int(latest['length'] - df.iloc[-2]['length'])}" if len(df) > 1 else None
                )

            with col3:
                success_rate = latest.get('success_rate', 0)
                st.metric(
                    label="Success Rate",
                    value=f"{success_rate:.1f}%"
                )

            # Display plots
            st.plotly_chart(
                self.create_metrics_plot(df),
                width='stretch'
            )

            # Display visualizations
            st.markdown("---")
            st.subheader("📸 Live Visualizations")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                st.write("**Training Progress**")
                live_viz_path = self.log_dir / 'live_training_viz.png'
                if live_viz_path.exists():
                    st.image(str(live_viz_path), width='stretch',
                            caption="Live Training Metrics (updates every 10 episodes)")
                else:
                    st.info("Live visualization will appear here when --live-viz is enabled")

            with viz_col2:
                st.write("**Current Training Frame**")
                render_path = self.log_dir / 'render_current.png'
                if render_path.exists():
                    st.image(str(render_path), width='stretch',
                            caption="Current maze state (updates during training with --render)")
                else:
                    st.info("Render frame will appear here when --render is enabled")

            # Statistics table
            st.subheader("📊 Training Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Overall Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Total Episodes', 'Average Reward', 'Best Reward',
                              'Average Length', 'Min Length'],
                    'Value': [
                        str(len(df)),
                        f"{df['reward'].mean():.2f}",
                        f"{df['reward'].max():.2f}",
                        f"{df['length'].mean():.1f}",
                        str(int(df['length'].min()))
                    ]
                })
                st.dataframe(stats_df, hide_index=True)

            with col2:
                st.write("**Recent Performance (Last 50 episodes)**")
                recent = df.tail(50)
                recent_stats_df = pd.DataFrame({
                    'Metric': ['Avg Reward', 'Avg Length', 'Success Rate'],
                    'Value': [
                        f"{recent['reward'].mean():.2f}",
                        f"{recent['length'].mean():.1f}",
                        f"{recent.get('success_rate', pd.Series([0])).iloc[-1]:.1f}%"
                    ]
                })
                st.dataframe(recent_stats_df, hide_index=True)

            # Recent episodes table
            st.subheader("📜 Recent Episodes")
            recent_episodes = df.tail(10)[['episode', 'reward', 'length']].copy()
            recent_episodes = recent_episodes.sort_values('episode', ascending=False)
            st.dataframe(recent_episodes, hide_index=True, width='stretch')

        else:
            st.info("No training data available yet. Start training to see metrics!")

            # Still show visualization slots even without training data
            st.markdown("---")
            st.subheader("📸 Live Visualizations")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                st.write("**Training Progress**")
                live_viz_path = self.log_dir / 'live_training_viz.png'
                if live_viz_path.exists():
                    st.image(str(live_viz_path), width='stretch',
                            caption="Live Training Metrics (updates every 10 episodes)")
                else:
                    st.info("Live visualization will appear here when training with --live-viz")

            with viz_col2:
                st.write("**Current Training Frame**")
                render_path = self.log_dir / 'render_current.png'
                if render_path.exists():
                    st.image(str(render_path), width='stretch',
                            caption="Current maze state (updates during training with --render)")
                else:
                    st.info("Render frame will appear here when training with --render")

    def render_demo_dashboard(self):
        """Render the demo/evaluation dashboard page."""
        st.title("🎮 Demo & Evaluation Dashboard")
        st.markdown("---")

        # Check for demo outputs
        demo_current = self.demo_dir / 'demo_current.png'
        demo_summary = self.demo_dir / 'demo_summary.png'
        demo_episodes = list(self.demo_dir.glob('demo_episode_*.pkl'))

        if not demo_current.exists() and not demo_summary.exists():
            st.info("No demo data available yet. Run a demo to see visualizations!")
            st.markdown("**Run a demo:**")
            st.code("python demo.py --model ./models/dqn_final.pth --episodes 10", language="bash")
            st.caption("Add --record flag to save episodes for replay")
            return

        # Current demo frame
        if demo_current.exists():
            st.subheader("🎯 Current Demo State")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(str(demo_current), width='stretch',
                        caption="Current demo frame (updates during demo)")

            with col2:
                st.markdown("**Demo Info**")
                if demo_episodes:
                    st.metric("Recorded Episodes", len(demo_episodes))
                st.info("This shows the agent's current position during demo playback")

        # Demo summary
        if demo_summary.exists():
            st.markdown("---")
            st.subheader("📈 Demo Performance Summary")
            st.image(str(demo_summary), width='stretch',
                    caption="Performance analysis across all demo episodes")

        # Recorded episodes
        if demo_episodes:
            st.markdown("---")
            st.subheader("🎬 Recorded Demo Episodes")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**{len(demo_episodes)} episodes available for replay**")

                # List episodes
                episode_names = sorted([ep.name for ep in demo_episodes])
                for ep_name in episode_names[:10]:  # Show first 10
                    st.text(f"• {ep_name}")

                if len(episode_names) > 10:
                    st.caption(f"...and {len(episode_names) - 10} more")

            with col2:
                st.markdown("**Replay an episode:**")
                st.code("python replay_viewer.py demo_outputs/demo_episode_0.pkl --export-images frames/",
                       language="bash")
                st.caption("This will generate an interactive HTML viewer")

        # Demo commands reference
        st.markdown("---")
        st.subheader("💡 Demo Commands")

        cmd_col1, cmd_col2 = st.columns(2)

        with cmd_col1:
            st.markdown("**Run basic demo:**")
            st.code("python demo.py --episodes 10", language="bash")

            st.markdown("**Demo with recording:**")
            st.code("python demo.py --episodes 10 --record", language="bash")

        with cmd_col2:
            st.markdown("**Custom output directory:**")
            st.code("python demo.py --save-dir my_demo --episodes 5", language="bash")

            st.markdown("**View recorded episode:**")
            st.code("python replay_viewer.py demo_outputs/demo_episode_0.pkl --summary", language="bash")

    def render_episode_viewer(self):
        """Render the episode viewer page."""
        st.title("🎬 Episode Viewer")
        st.markdown("---")

        viewer_path = Path("viewer.html")

        if not viewer_path.exists():
            st.info("No episode viewer available yet.")
            st.markdown("**To generate an episode viewer:**")
            st.markdown("1. Record episodes during training or demo")
            st.code("python demo.py --episodes 10 --record", language="bash")
            st.markdown("2. Export frames to create the viewer")
            st.code("python replay_viewer.py demo_outputs/demo_episode_0.pkl --export-images frames/", language="bash")
            st.caption("This will generate a viewer.html file for interactive playback")
            return

        # Display the episode viewer
        st.subheader("🎥 Interactive Episode Playback")

        # Read and embed the viewer
        with open(viewer_path, 'r') as f:
            viewer_html = f.read()

        st.components.v1.html(viewer_html, height=800, scrolling=True)

        # Additional information
        st.markdown("---")
        st.subheader("ℹ️ Viewer Controls")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Navigation:**")
            st.markdown("- Use the frame slider to navigate")
            st.markdown("- Click on specific frames for details")
            st.markdown("- View action history and rewards")

        with col2:
            st.markdown("**Episode Data:**")
            st.markdown("- Frame-by-frame annotations")
            st.markdown("- Cumulative rewards")
            st.markdown("- Agent actions at each step")


def main():
    """Main entry point for dashboard."""
    dashboard = TrainingDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()
