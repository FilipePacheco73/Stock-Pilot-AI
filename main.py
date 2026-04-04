"""
Main training and evaluation script for StockPilot
"""

import numpy as np
import os
import pickle
from pathlib import Path
from src.environment import SupplyChainEnv
from src.agents import BaselinePolicy, run_baseline_simulation, RLTrainer
from src.utils import SimulationMetrics, create_simulation_dataframe


def test_environment():
    """Test that the environment works correctly."""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT")
    print("="*60)
    
    env = SupplyChainEnv(seed=42)
    obs, info = env.reset()
    
    print(f"✓ Environment created successfully")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    
    # Run 10 steps
    print(f"\nRunning 10 test steps...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: Demand={info['demand']:.1f}, Inventory={info['inventory']:.1f}, "
              f"Service={info['service_level']:.2%}")
    
    print(f"\n✓ Environment test passed!")
    return env


def run_baseline(env: SupplyChainEnv):
    """Run baseline policy simulation."""
    print("\n" + "="*60)
    print("BASELINE POLICY EVALUATION")
    print("="*60)
    
    # Create baseline policy
    baseline = BaselinePolicy(
        safety_stock=75,
        reorder_point=75,
        lead_time_estimate=3.0,
        demand_mean=50.0,
    )
    
    print(f"Policy: {baseline.get_name()}")
    print(f"\nRunning baseline simulation (365 days)...")
    
    # Run baseline
    results = run_baseline_simulation(env, baseline, episodes=1, episode_length=365)
    
    # Get metrics
    metrics = SimulationMetrics.compute_metrics(env)
    
    print(f"\n✓ Baseline Simulation Complete:")
    print(f"  Total Cost: ${metrics['total_cost']:.2f}")
    print(f"  Service Level: {metrics['service_level']:.2%}")
    print(f"  Stockout Events: {metrics['stockout_events']}")
    print(f"  Avg Inventory: {metrics['avg_inventory']:.1f}")
    print(f"  Avg Daily Cost: ${metrics['avg_daily_cost']:.2f}")
    
    # Save baseline metrics
    baseline_metrics = metrics
    
    return baseline_metrics, results


def train_agent(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
):
    """Train RL agent."""
    print("\n" + "="*60)
    print("RL AGENT TRAINING")
    print("="*60)
    
    # Create trainer
    trainer = RLTrainer(
        model_dir="models",
        log_dir="logs",
        seed=42,
    )
    
    print(f"\nTraining configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Algorithm: PPO")
    
    try:
        # Train agent
        model, info = trainer.train(
            total_timesteps=total_timesteps,
            n_envs=n_envs,
        )
        
        print(f"\n✓ Training Complete!")
        print(f"  Model saved to: {info['model_path']}")
        
        return model, trainer
    
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print(f"\nTip: Make sure you have installed PyTorch and CUDA drivers if using GPU.")
        print(f"You can install PyTorch separately: pip install torch torchvision torchaudio")
        return None, trainer


def evaluate_agent(
    model,
    trainer: RLTrainer,
    n_episodes: int = 5,
):
    """Evaluate trained agent."""
    print("\n" + "="*60)
    print("RL AGENT EVALUATION")
    print("="*60)
    
    if model is None:
        print("✗ No trained model available. Skipping evaluation.")
        return None
    
    print(f"\nEvaluating agent over {n_episodes} episodes...")
    
    results = trainer.evaluate(n_episodes=n_episodes, deterministic=True)
    
    print(f"\n✓ Evaluation Complete:")
    print(f"  Avg Total Cost: ${results['avg_total_cost']:.2f} ± ${results['std_total_cost']:.2f}")
    print(f"  Service Level: {results['avg_service_level']:.2%}")
    print(f"  Avg Stockouts: {results['avg_stockouts']:.1f}")
    print(f"  Avg Inventory: {results['avg_inventory']:.1f}")
    
    return results


def compare_and_save(baseline_metrics, rl_results):
    """Compare policies and save results."""
    print("\n" + "="*60)
    print("COMPARISON: BASELINE vs RL AGENT")
    print("="*60)
    
    if rl_results is None:
        print("✗ RL agent evaluation not available. Skipping comparison.")
        return
    
    # Create RL metrics dict from results
    rl_metrics = {
        "total_cost": rl_results["avg_total_cost"],
        "service_level": rl_results["avg_service_level"],
        "stockout_events": rl_results["avg_stockouts"],
        "avg_inventory": rl_results["avg_inventory"],
    }
    
    comparison = SimulationMetrics.compare_policies(baseline_metrics, rl_metrics)
    
    improvements = comparison["improvements"]
    
    print(f"\n📊 Key Improvements (RL vs Baseline):")
    if "cost_reduction_pct" in improvements:
        print(f"  ✓ Cost Reduction: {improvements['cost_reduction_pct']:.1f}%")
        print(f"    (${improvements['cost_reduction']:.2f} savings per episode)")
    if "service_level_improvement" in improvements:
        print(f"  ✓ Service Level: +{improvements['service_level_improvement']:.2%}")
    if "stockout_reduction" in improvements:
        print(f"  ✓ Fewer Stockouts: {improvements['stockout_reduction']:.0f} fewer events")
    
    print(f"\n📈 Category Breakdown:")
    print(f"  Baseline:")
    print(f"    - Total Cost: ${baseline_metrics['total_cost']:.2f}")
    print(f"    - Service Level: {baseline_metrics['service_level']:.2%}")
    
    print(f"\n  RL Agent:")
    print(f"    - Total Cost: ${rl_metrics['total_cost']:.2f}")
    print(f"    - Service Level: {rl_metrics['service_level']:.2%}")
    
    # Save comparison
    os.makedirs("data", exist_ok=True)
    comparison_path = "data/comparison.pkl"
    
    with open(comparison_path, "wb") as f:
        pickle.dump(comparison, f)
    
    print(f"\n✓ Comparison saved to: {comparison_path}")


def main(
    train_agent_flag: bool = True,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
):
    """
    Main training and evaluation pipeline.
    
    Args:
        train_agent_flag: Whether to train RL agent
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
    """
    print("\n" + "="*60)
    print("STOCKPILOT: RL Supply Chain Optimization")
    print("="*60)
    
    # Test environment
    env = test_environment()
    
    # Run baseline
    baseline_metrics, baseline_results = run_baseline(env)
    
    # Train agent
    if train_agent_flag:
        model, trainer = train_agent(
            total_timesteps=total_timesteps,
            n_envs=n_envs,
        )
        
        # Evaluate agent
        if model is not None:
            rl_results = evaluate_agent(model, trainer, n_episodes=5)
            compare_and_save(baseline_metrics, rl_results)
    else:
        print("\nSkipping agent training (train_agent_flag=False)")
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: streamlit run dashboard/app.py")
    print("2. View interactive visualizations and comparisons")
    print("\n")


if __name__ == "__main__":
    # Run with full training
    main(
        train_agent_flag=True,
        total_timesteps=500_000,
        n_envs=4,
    )
