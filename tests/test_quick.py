"""
Quick test script for StockPilot (without training)
This tests environment and baseline policy without expensive RL training
"""

import numpy as np
from src.environment import SupplyChainEnv
from src.agents import BaselinePolicy, run_baseline_simulation
from src.utils import SimulationMetrics


def test_quick():
    """Quick test of environment and baseline."""
    print("\n" + "="*60)
    print("STOCKPILOT QUICK TEST")
    print("="*60)
    
    # Test 1: Environment
    print("\n[1/3] Testing Environment...")
    env = SupplyChainEnv(
        initial_inventory=100,
        holding_cost=0.5,
        stockout_cost=10.0,
        ordering_cost=5.0,
        episode_length=365,
        seed=42,
    )
    
    obs, info = env.reset()
    print(f"✓ Environment created")
    print(f"  Initial inventory: {obs[0]:.1f}")
    print(f"  Initial safety stock: {obs[1]:.1f}")
    
    # Run a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✓ Environment steps executed successfully")
    
    # Test 2: Baseline Policy
    print("\n[2/3] Testing Baseline Policy...")
    env.reset()
    
    baseline = BaselinePolicy(
        safety_stock=75,
        reorder_point=75,
        lead_time_estimate=3.0,
        demand_mean=50.0,
    )
    
    print(f"✓ Baseline policy created: {baseline.get_name()}")
    
    # Run full simulation
    print("\nRunning 365-day baseline simulation...")
    results = run_baseline_simulation(env, baseline, episodes=1, episode_length=365)
    
    # Get metrics
    metrics = SimulationMetrics.compute_metrics(env)
    
    print(f"✓ Baseline simulation complete")
    print(f"\n📊 Results:")
    print(f"  Total Cost: ${metrics['total_cost']:.2f}")
    print(f"  Service Level: {metrics['service_level']:.2%}")
    print(f"  Stockouts: {metrics['stockout_events']}")
    print(f"  Avg Inventory: {metrics['avg_inventory']:.1f}")
    
    # Test 3: Random Policy Comparison
    print("\n[3/3] Comparing with Random Policy...")
    env.reset()
    
    total_reward = 0
    for step in range(365):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    random_metrics = SimulationMetrics.compute_metrics(env)
    
    print(f"✓ Random policy evaluation complete")
    print(f"\n📈 Comparison:")
    print(f"  Baseline Total Cost: ${metrics['total_cost']:.2f}")
    print(f"  Random Policy Cost:  ${random_metrics['total_cost']:.2f}")
    cost_diff = metrics['total_cost'] - random_metrics['total_cost']
    print(f"  Difference: ${abs(cost_diff):.2f} (Baseline {'wins' if cost_diff < 0 else 'loses'})")
    
    print("\n" + "="*60)
    print("✓ QUICK TEST PASSED!")
    print("="*60)
    print("\nYou can now:")
    print("• Run full training: python main.py")
    print("• View dashboard: streamlit run dashboard/app.py")
    print("\n")


if __name__ == "__main__":
    test_quick()
