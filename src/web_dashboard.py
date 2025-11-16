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

    def __init__(self, log_dir: str = './outputs', model_dir: str = './models'):
        """
        Initialize dashboard.

        Args:
            log_dir: Directory for training logs
            model_dir: Directory for saved models
        """
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)

        # Create directories if needed
        self.log_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

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

        st.title("🧩 Visual Maze Solving - Training Dashboard")
        st.markdown("---")

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")

            # Refresh rate
            refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 5)

            # Auto-refresh toggle
            auto_refresh = st.checkbox("Auto-refresh", value=True)

            st.markdown("---")

            # Episode viewer link
            st.subheader("🎬 Episode Viewer")
            viewer_path = Path("viewer.html")
            if viewer_path.exists():
                st.markdown("**[Open Episode Viewer](viewer.html)** 📺")
                st.caption("View recorded episodes interactively")
            else:
                st.info("Export episode frames to generate viewer.html")

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

            # Manual refresh button
            if st.button("🔄 Refresh Now"):
                st.rerun()

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

        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()


def main():
    """Main entry point for dashboard."""
    dashboard = TrainingDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()
