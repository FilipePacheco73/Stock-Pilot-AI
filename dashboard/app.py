"""
StockPilot Interactive Dashboard
Streamlit app for supply chain simulation visualization and RL agent comparison
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SupplyChainEnv
from src.agents import BaselinePolicy, run_baseline_simulation
from src.utils import SimulationMetrics, create_simulation_dataframe, get_cost_breakdown

# Page configuration
st.set_page_config(
    page_title="StockPilot - Supply Chain RL Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
        }
        .header-title {
            color: #667eea;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subheader {
            color: #555;
            font-size: 1.5em;
            margin: 20px 0 10px 0;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Load trained RL model if it exists."""
    model_path = Path("models/ppo_agent_final.zip")
    if model_path.exists():
        try:
            from stable_baselines3 import PPO
            return PPO.load(str(model_path))
        except Exception as e:
            st.warning(f"Could not load model: {e}")
            return None
    return None


@st.cache_data
def run_simulation(env_seed: int, use_baseline: bool = True, use_rl: bool = True):
    """Run simulation with both policies."""
    results = {}
    
    # Baseline simulation
    if use_baseline:
        env = SupplyChainEnv(seed=env_seed)
        baseline = BaselinePolicy(
            safety_stock=75,
            reorder_point=75,
            lead_time_estimate=3.0,
            demand_mean=50.0,
        )
        
        obs, _ = env.reset()
        baseline_data = {
            "demands": [],
            "inventory": [],
            "safety_stock": [],
            "fulfilled": [],
            "costs": [],
            "orders": [],
        }
        
        for step in range(365):
            inventory = obs[0]
            pipeline_qty = obs[2]
            demand_history = env.demand_history
            
            action_dict = baseline.decide(inventory, pipeline_qty, demand_history)
            action = np.array([action_dict["safety_stock_adj"], action_dict["order_qty"]], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            baseline_data["demands"].append(info["demand"])
            baseline_data["inventory"].append(info["inventory"])
            baseline_data["safety_stock"].append(env.safety_stock)
            baseline_data["fulfilled"].append(info["fulfilled_demand"])
            cost = info["holding_cost"] + info["stockout_cost"] + info["ordering_cost"]
            baseline_data["costs"].append(cost)
            
            if action_dict["order_qty"] > 0:
                baseline_data["orders"].append(step)
        
        baseline_metrics = SimulationMetrics.compute_metrics(env)
        results["baseline"] = {
            "data": baseline_data,
            "metrics": baseline_metrics,
            "env": env,
        }
    
    # RL simulation
    if use_rl:
        model = load_trained_model()
        
        env = SupplyChainEnv(seed=env_seed)
        obs, _ = env.reset()
        rl_data = {
            "demands": [],
            "inventory": [],
            "safety_stock": [],
            "fulfilled": [],
            "costs": [],
            "orders": [],
        }
        
        for step in range(365):
            if model is not None:
                # Use trained RL agent
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Use heuristic if no model
                inventory = obs[0]
                pipeline_qty = obs[2]
                avg_demand = np.mean(obs[3:4]) if len(obs) > 3 else 50
                
                # Simple heuristic: adjust safety stock based on inventory levels
                if inventory < avg_demand:
                    action = np.array([5.0, 200.0], dtype=np.float32)
                elif inventory > avg_demand * 2:
                    action = np.array([-5.0, 0.0], dtype=np.float32)
                else:
                    action = np.array([0.0, 0.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            rl_data["demands"].append(info["demand"])
            rl_data["inventory"].append(info["inventory"])
            rl_data["safety_stock"].append(env.safety_stock)
            rl_data["fulfilled"].append(info["fulfilled_demand"])
            cost = info["holding_cost"] + info["stockout_cost"] + info["ordering_cost"]
            rl_data["costs"].append(cost)
            
            # Track orders
            if action[1] > 0.5:
                rl_data["orders"].append(step)
        
        rl_metrics = SimulationMetrics.compute_metrics(env)
        results["rl"] = {
            "data": rl_data,
            "metrics": rl_metrics,
            "env": env,
        }
    
    return results


def plot_inventory_comparison(results):
    """Plot inventory levels for both policies."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Baseline Policy", "RL Agent"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )
    
    # Baseline
    if "baseline" in results:
        baseline_data = results["baseline"]["data"]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(baseline_data["inventory"]))),
                y=baseline_data["inventory"],
                name="Inventory (Baseline)",
                line=dict(color="#1f77b4", width=2),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.2)",
            ),
            row=1, col=1,
        )
        
        # Safety stock overlay
        fig.add_trace(
            go.Scatter(
                x=list(range(len(baseline_data["safety_stock"]))),
                y=baseline_data["safety_stock"],
                name="Safety Stock (Baseline)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            ),
            row=1, col=1,
        )
    
    # RL
    if "rl" in results:
        rl_data = results["rl"]["data"]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rl_data["inventory"]))),
                y=rl_data["inventory"],
                name="Inventory (RL)",
                line=dict(color="#2ca02c", width=2),
                fill="tozeroy",
                fillcolor="rgba(44, 160, 44, 0.2)",
            ),
            row=1, col=2,
        )
        
        # Safety stock overlay
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rl_data["safety_stock"]))),
                y=rl_data["safety_stock"],
                name="Safety Stock (RL)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            ),
            row=1, col=2,
        )
    
    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_xaxes(title_text="Day", row=1, col=2)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Units", row=1, col=2)
    
    fig.update_layout(height=500, hovermode="x unified", showlegend=True)
    return fig


def plot_demand_fulfillment(results):
    """Plot demand vs fulfilled demand."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Baseline Policy", "RL Agent"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )
    
    # Baseline
    if "baseline" in results:
        baseline_data = results["baseline"]["data"]
        days = list(range(len(baseline_data["demands"])))
        
        fig.add_trace(
            go.Bar(
                x=days,
                y=baseline_data["demands"],
                name="Demand",
                marker=dict(color="rgba(31, 119, 180, 0.5)"),
                showlegend=True,
            ),
            row=1, col=1,
        )
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=baseline_data["fulfilled"],
                name="Fulfilled",
                line=dict(color="#2ca02c", width=3),
                showlegend=True,
            ),
            row=1, col=1,
        )
    
    # RL
    if "rl" in results:
        rl_data = results["rl"]["data"]
        days = list(range(len(rl_data["demands"])))
        
        fig.add_trace(
            go.Bar(
                x=days,
                y=rl_data["demands"],
                name="Demand",
                marker=dict(color="rgba(31, 119, 180, 0.5)"),
                showlegend=False,
            ),
            row=1, col=2,
        )
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=rl_data["fulfilled"],
                name="Fulfilled",
                line=dict(color="#2ca02c", width=3),
                showlegend=False,
            ),
            row=1, col=2,
        )
    
    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_xaxes(title_text="Day", row=1, col=2)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Units", row=1, col=2)
    
    fig.update_layout(height=500, hovermode="x unified")
    return fig


def plot_cumulative_cost(results):
    """Plot cumulative costs over time."""
    fig = go.Figure()
    
    if "baseline" in results:
        baseline_data = results["baseline"]["data"]
        cumulative_cost = np.cumsum(baseline_data["costs"])
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_cost))),
                y=cumulative_cost,
                name="Baseline",
                line=dict(color="#1f77b4", width=3),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.2)",
            )
        )
    
    if "rl" in results:
        rl_data = results["rl"]["data"]
        cumulative_cost = np.cumsum(rl_data["costs"])
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_cost))),
                y=cumulative_cost,
                name="RL Agent",
                line=dict(color="#2ca02c", width=3),
                fill="tozeroy",
                fillcolor="rgba(44, 160, 44, 0.2)",
            )
        )
    
    fig.update_layout(
        title="Cumulative Cost Over Time",
        xaxis_title="Day",
        yaxis_title="Cumulative Cost ($)",
        height=500,
        hovermode="x unified",
    )
    return fig


def plot_cost_breakdown(results):
    """Plot cost breakdown for both policies."""
    categories = ["Holding", "Stockout", "Ordering"]
    
    baseline_values = []
    rl_values = []
    
    if "baseline" in results:
        metrics = results["baseline"]["metrics"]
        baseline_values = [
            metrics.get("holding_cost_total", 0),
            metrics.get("stockout_cost_total", 0),
            metrics.get("ordering_cost_total", 0),
        ]
    
    if "rl" in results:
        metrics = results["rl"]["metrics"]
        rl_values = [
            metrics.get("holding_cost_total", 0),
            metrics.get("stockout_cost_total", 0),
            metrics.get("ordering_cost_total", 0),
        ]
    
    fig = go.Figure(data=[
        go.Bar(name="Baseline", x=categories, y=baseline_values, marker=dict(color="#1f77b4")),
        go.Bar(name="RL Agent", x=categories, y=rl_values, marker=dict(color="#2ca02c")),
    ])
    
    fig.update_layout(
        title="Cost Breakdown Comparison",
        yaxis_title="Cost ($)",
        barmode="group",
        height=500,
    )
    return fig


def plot_metrics_radar(results):
    """Plot metrics comparison as radar chart."""
    categories = ["Service Level", "Avg Inventory", "Stockouts (inverted)"]
    
    baseline_metrics = results.get("baseline", {}).get("metrics", {})
    rl_metrics = results.get("rl", {}).get("metrics", {})
    
    # Normalize metrics (0-100 scale)
    baseline_values = [
        baseline_metrics.get("service_level", 0) * 100,  # 0-100%
        100 - min(baseline_metrics.get("avg_inventory", 0) / 2, 100),  # Lower is better
        min(100 - baseline_metrics.get("stockout_events", 100), 100),  # Fewer is better
    ]
    
    rl_values = [
        rl_metrics.get("service_level", 0) * 100,
        100 - min(rl_metrics.get("avg_inventory", 0) / 2, 100),
        min(100 - rl_metrics.get("stockout_events", 100), 100),
    ]
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=baseline_values,
            theta=categories,
            name="Baseline",
            fill="toself",
            line=dict(color="#1f77b4"),
        ),
        go.Scatterpolar(
            r=rl_values,
            theta=categories,
            name="RL Agent",
            fill="toself",
            line=dict(color="#2ca02c"),
        ),
    ])
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500,
    )
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1 class='header-title'>🚀 StockPilot</h1>", unsafe_allow_html=True)
        st.markdown("**AI-Powered Supply Chain Optimization Dashboard**")
    with col2:
        st.markdown("")
        st.markdown("")
        model_status = "✅ Loaded" if load_trained_model() is not None else "⚠️ Mock"
        st.metric("Model Status", model_status)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        env_seed = st.slider("Random Seed", 0, 1000, 42, step=1)
        
        st.subheader("Simulation Options")
        run_baseline = st.checkbox("Run Baseline Policy", value=True)
        run_rl = st.checkbox("Run RL Agent", value=True)
        
        if not run_baseline and not run_rl:
            st.error("Please select at least one policy to simulate!")
        
        st.markdown("---")
        st.subheader("Environment Parameters")
        with st.expander("Demand Settings"):
            demand_mean = st.slider("Mean Demand (units/day)", 20, 100, 50)
            demand_std = st.slider("Demand Std Dev", 5, 30, 15)
            spike_prob = st.slider("Spike Probability", 0.0, 0.5, 0.1, step=0.01)
        
        with st.expander("Cost Settings"):
            holding_cost = st.slider("Holding Cost ($/unit/day)", 0.1, 2.0, 0.5, step=0.1)
            stockout_cost = st.slider("Stockout Cost ($/unit)", 5.0, 20.0, 10.0, step=1.0)
            ordering_cost = st.slider("Ordering Cost ($)", 1.0, 10.0, 5.0, step=1.0)
        
        st.markdown("---")
        if st.button("🔄 Run Simulation", use_container_width=True):
            st.session_state.run_simulation = True
    
    # Main content
    if "run_simulation" not in st.session_state:
        st.session_state.run_simulation = True
    
    if st.session_state.run_simulation and (run_baseline or run_rl):
        with st.spinner("Running simulation..."):
            results = run_simulation(env_seed, use_baseline=run_baseline, use_rl=run_rl)
        
        # Key metrics row
        st.markdown("<h2 class='subheader'>📊 Key Metrics</h2>", unsafe_allow_html=True)
        
        metrics_cols = st.columns(2 if (run_baseline and run_rl) else 1)
        
        if run_baseline and "baseline" in results:
            with metrics_cols[0]:
                baseline_metrics = results["baseline"]["metrics"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Total Cost",
                        f"${baseline_metrics['total_cost']:.2f}",
                        delta=None,
                    )
                    st.metric(
                        "Service Level",
                        f"{baseline_metrics['service_level']:.1%}",
                        delta=None,
                    )
                with col2:
                    st.metric(
                        "Stockout Events",
                        f"{baseline_metrics['stockout_events']:.0f}",
                        delta=None,
                    )
                    st.metric(
                        "Avg Inventory",
                        f"{baseline_metrics['avg_inventory']:.1f} units",
                        delta=None,
                    )
                
                st.markdown("**Baseline Policy**")
        
        if run_rl and "rl" in results:
            with metrics_cols[1 if (run_baseline and run_rl) else 0]:
                rl_metrics = results["rl"]["metrics"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Total Cost",
                        f"${rl_metrics['total_cost']:.2f}",
                        delta=None,
                    )
                    st.metric(
                        "Service Level",
                        f"{rl_metrics['service_level']:.1%}",
                        delta=None,
                    )
                with col2:
                    st.metric(
                        "Stockout Events",
                        f"{rl_metrics['stockout_events']:.0f}",
                        delta=None,
                    )
                    st.metric(
                        "Avg Inventory",
                        f"{rl_metrics['avg_inventory']:.1f} units",
                        delta=None,
                    )
                
                st.markdown("**RL Agent**")
        
        # Improvements section
        if run_baseline and run_rl and "baseline" in results and "rl" in results:
            st.markdown("---")
            st.markdown("<h2 class='subheader'>🎯 RL Agent Improvements</h2>", unsafe_allow_html=True)
            
            baseline_metrics = results["baseline"]["metrics"]
            rl_metrics = results["rl"]["metrics"]
            
            comparison = SimulationMetrics.compare_policies(baseline_metrics, rl_metrics)
            improvements = comparison["improvements"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cost_reduction_pct = improvements.get("cost_reduction_pct", 0)
                st.metric(
                    "💰 Cost Reduction",
                    f"{cost_reduction_pct:.1f}%",
                    delta=f"${improvements.get('cost_reduction', 0):.2f}",
                )
            
            with col2:
                service_improvement = improvements.get("service_level_improvement", 0)
                st.metric(
                    "📈 Service Level",
                    f"{service_improvement:+.1%}",
                    delta="Better" if service_improvement > 0 else "Worse",
                )
            
            with col3:
                stockout_reduction = improvements.get("stockout_reduction", 0)
                st.metric(
                    "✅ Fewer Stockouts",
                    f"{stockout_reduction:.0f} events",
                    delta="Fewer" if stockout_reduction > 0 else "More",
                )
        
        # Visualizations
        st.markdown("---")
        st.markdown("<h2 class='subheader'>📈 Visualizations</h2>", unsafe_allow_html=True)
        
        # Inventory comparison
        if run_baseline or run_rl:
            st.subheader("Inventory Levels Over Time")
            st.plotly_chart(plot_inventory_comparison(results), use_container_width=True)
        
        # Demand fulfillment
        if run_baseline or run_rl:
            st.subheader("Demand vs Fulfilled Demand")
            st.plotly_chart(plot_demand_fulfillment(results), use_container_width=True)
        
        # Cost comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cumulative Cost")
            st.plotly_chart(plot_cumulative_cost(results), use_container_width=True)
        
        with col2:
            st.subheader("Cost Breakdown")
            st.plotly_chart(plot_cost_breakdown(results), use_container_width=True)
        
        # Radar chart
        if run_baseline and run_rl and "baseline" in results and "rl" in results:
            st.subheader("Multi-Metric Comparison")
            st.plotly_chart(plot_metrics_radar(results), use_container_width=True)
        
        # Detailed data
        st.markdown("---")
        st.markdown("<h2 class='subheader'>📋 Detailed Results</h2>", unsafe_allow_html=True)
        
        if run_baseline and "baseline" in results:
            with st.expander("Baseline Policy Details"):
                df = create_simulation_dataframe(results["baseline"]["env"])
                st.dataframe(df.head(20), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Full Statistics**")
                    st.json({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in results["baseline"]["metrics"].items()})
        
        if run_rl and "rl" in results:
            with st.expander("RL Agent Details"):
                df = create_simulation_dataframe(results["rl"]["env"])
                st.dataframe(df.head(20), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Full Statistics**")
                    st.json({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in results["rl"]["metrics"].items()})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>StockPilot v0.1.0 | AI-Powered Supply Chain Optimization</p>
        <p>Built with Streamlit, Stable-Baselines3, and Gymnasium</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
