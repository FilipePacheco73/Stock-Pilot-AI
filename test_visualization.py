"""
Simple test to validate training and visualization with matplotlib.
This bypasses Streamlit to ensure core logic works.
Saves all results with timestamp for easy tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.environment.supply_chain_env import SupplyChainEnv
from src.agents.baseline_policy import BaselinePolicy
from src.utils.metrics import SimulationMetrics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class ScheduledCostSupplyChainEnv(SupplyChainEnv):
    """Environment wrapper that randomizes cost parameters every 30 days per episode."""

    def __init__(self, schedule_window=30, base_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.schedule_window = schedule_window
        self.base_seed = base_seed
        self.episode_idx = 0
        self.h_sched = None
        self.s_sched = None
        self.o_sched = None

    def _apply_day_costs(self):
        if self.h_sched is None:
            return
        day_idx = min(self.current_step, len(self.h_sched) - 1)
        self.holding_cost = float(self.h_sched[day_idx])
        self.stockout_cost = float(self.s_sched[day_idx])
        self.ordering_cost = float(self.o_sched[day_idx])

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        episode_seed = self.base_seed + self.episode_idx
        self.h_sched, self.s_sched, self.o_sched = generate_cost_schedule(
            self.episode_length,
            seed=episode_seed,
            window=self.schedule_window,
        )
        self.episode_idx += 1
        self._apply_day_costs()
        return self._get_state(), info

    def step(self, action):
        self._apply_day_costs()
        return super().step(action)


def generate_cost_schedule(days, seed, window=30):
    """
    Pre-generate cost parameter values that change every 30 simulation days.
    Both Baseline and RL receive the same schedule so the comparison is fair.
    """
    rng = np.random.RandomState(seed + 9999)
    n_windows = (days + window - 1) // window
    holding_vals, stockout_vals, ordering_vals = [], [], []
    for _ in range(n_windows):
        holding_vals.append(round(rng.uniform(0.1, 1.0), 3))
        stockout_vals.append(round(rng.uniform(5.0, 15.0), 3))
        ordering_vals.append(round(rng.uniform(2.0, 10.0), 3))
    # Expand to per-day arrays
    h = np.repeat(holding_vals,  window)[:days]
    s = np.repeat(stockout_vals, window)[:days]
    o = np.repeat(ordering_vals, window)[:days]
    return h, s, o


def run_baseline_period(seed, days, cost_schedule=None, env_kwargs=None):
    """Run baseline policy for a fixed number of days."""
    env = SupplyChainEnv(seed=seed, **(env_kwargs or {}))
    baseline = BaselinePolicy(
        safety_stock=300,
        reorder_point=300,
        lead_time_estimate=3.0,
        demand_mean=50.0,
    )
    obs, _ = env.reset()
    data = {
        "inventory": [],
        "safety_stock": [],
        "costs": [],
        "holding_costs": [],
        "stockout_costs": [],
        "ordering_costs": [],
        "demands": [],
        "fulfilled": [],
    }

    h_sched, s_sched, o_sched = cost_schedule if cost_schedule is not None else (None, None, None)

    data["holding_param"] = []
    data["stockout_param"] = []
    data["ordering_param"] = []

    for step in range(days):
        if h_sched is not None:
            env.holding_cost  = float(h_sched[step])
            env.stockout_cost = float(s_sched[step])
            env.ordering_cost = float(o_sched[step])
            obs = env._get_state()
        action_dict = baseline.decide(obs[0], obs[2], env.demand_history, current_safety_stock=obs[1])
        action = env.encode_action(action_dict["safety_stock_adj"])
        obs, _, _, _, info = env.step(action)
        data["inventory"].append(info["inventory"])
        data["safety_stock"].append(env.safety_stock)
        data["demands"].append(info["demand"])
        data["fulfilled"].append(info["fulfilled_demand"])
        data["holding_costs"].append(info["holding_cost"])
        data["stockout_costs"].append(info["stockout_cost"])
        data["ordering_costs"].append(info["ordering_cost"])
        data["holding_param"].append(env.holding_cost)
        data["stockout_param"].append(env.stockout_cost)
        data["ordering_param"].append(env.ordering_cost)
        cost = info["holding_cost"] + info["stockout_cost"] + info["ordering_cost"]
        data["costs"].append(cost)

    return SimulationMetrics.compute_metrics(env), data


def run_rl_period(model, seed, days, deterministic=True, cost_schedule=None, env_kwargs=None):
    """Run RL policy for a fixed number of days."""
    env = SupplyChainEnv(seed=seed, **(env_kwargs or {}))
    obs, _ = env.reset()
    data = {
        "inventory": [],
        "safety_stock": [],
        "costs": [],
        "holding_costs": [],
        "stockout_costs": [],
        "ordering_costs": [],
        "demands": [],
        "fulfilled": [],
    }

    h_sched, s_sched, o_sched = cost_schedule if cost_schedule is not None else (None, None, None)

    data["holding_param"] = []
    data["stockout_param"] = []
    data["ordering_param"] = []

    for step in range(days):
        if h_sched is not None:
            env.holding_cost  = float(h_sched[step])
            env.stockout_cost = float(s_sched[step])
            env.ordering_cost = float(o_sched[step])
            obs = env._get_state()
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, _, _, info = env.step(action)
        data["inventory"].append(info["inventory"])
        data["safety_stock"].append(env.safety_stock)
        data["demands"].append(info["demand"])
        data["fulfilled"].append(info["fulfilled_demand"])
        data["holding_costs"].append(info["holding_cost"])
        data["stockout_costs"].append(info["stockout_cost"])
        data["ordering_costs"].append(info["ordering_cost"])
        data["holding_param"].append(env.holding_cost)
        data["stockout_param"].append(env.stockout_cost)
        data["ordering_param"].append(env.ordering_cost)
        cost = info["holding_cost"] + info["stockout_cost"] + info["ordering_cost"]
        data["costs"].append(cost)

    return SimulationMetrics.compute_metrics(env), data


def save_period_chart(results_dir, period_name, days, baseline_data, rl_data, output_name):
    """Save a dedicated chart for one period with a separate visual identity."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'{period_name} Period ({days} days): Baseline vs RL', fontsize=16, fontweight='bold')

    INV_COLORS = {
        "baseline_inventory": "#1f77b4",   # blue
        "baseline_safety":    "#d62728",   # red (dashed)
        "rl_inventory":       "#1f77b4",   # blue
        "rl_safety":          "#d62728",   # red (dashed)
    }

    x_axis = list(range(days))

    ax = axes[0, 0]
    ax.plot(x_axis, baseline_data["inventory"], label="Baseline Inventory", color=INV_COLORS["baseline_inventory"], linewidth=2)
    ax.axhline(y=300, color=INV_COLORS["baseline_safety"], linestyle="--", label="Baseline Safety Stock", linewidth=1.5)
    ax.set_title(f"{period_name}: Baseline Inventory", fontweight='bold')
    ax.set_xlabel("Day")
    ax.set_ylabel("Units")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x_axis, rl_data["inventory"], label="RL Inventory", color=INV_COLORS["rl_inventory"], linewidth=2)
    ax.plot(x_axis, rl_data["safety_stock"], label="RL Safety Stock", color=INV_COLORS["rl_safety"], linestyle="--", linewidth=2)
    ax.set_title(f"{period_name}: RL Inventory + Safety Stock", fontweight='bold')
    ax.set_xlabel("Day")
    ax.set_ylabel("Units")
    ax.legend()
    ax.grid(True, alpha=0.3)

    COST_COLORS = {
        "holding":  "#1f77b4",   # blue
        "stockout": "#d62728",   # red
        "ordering": "#2ca02c",   # green
    }

    b_holding = np.cumsum(baseline_data["holding_costs"])
    b_stockout = np.cumsum(baseline_data["stockout_costs"])
    b_ordering = np.cumsum(baseline_data["ordering_costs"])
    r_holding = np.cumsum(rl_data["holding_costs"])
    r_stockout = np.cumsum(rl_data["stockout_costs"])
    r_ordering = np.cumsum(rl_data["ordering_costs"])
    b_total = b_holding[-1] + b_stockout[-1] + b_ordering[-1]
    r_total = r_holding[-1] + r_stockout[-1] + r_ordering[-1]

    ax = axes[1, 0]
    ax.plot(x_axis, b_holding,  label="Holding",  color=COST_COLORS["holding"],  linewidth=2)
    ax.plot(x_axis, b_stockout, label="Stockout", color=COST_COLORS["stockout"], linewidth=2)
    ax.plot(x_axis, b_ordering, label="Ordering", color=COST_COLORS["ordering"], linewidth=2)
    ax.plot([], [], " ", label=f"Total = ${b_total:,.0f}")
    ax.set_title(f"{period_name}: Baseline Cumulative Cost Breakdown", fontweight='bold')
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_axis, r_holding,  label="Holding",  color=COST_COLORS["holding"],  linewidth=2)
    ax.plot(x_axis, r_stockout, label="Stockout", color=COST_COLORS["stockout"], linewidth=2)
    ax.plot(x_axis, r_ordering, label="Ordering", color=COST_COLORS["ordering"], linewidth=2)
    ax.plot([], [], " ", label=f"Total = ${r_total:,.0f}")
    ax.set_title(f"{period_name}: RL Cumulative Cost Breakdown", fontweight='bold')
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Row 3: cost parameter multipliers (step function, changes every 30 days) ---
    ax = axes[2, 0]
    ax.step(x_axis, baseline_data["holding_param"],  label="Holding coef",  color=COST_COLORS["holding"],  linewidth=2, where='post')
    ax.step(x_axis, baseline_data["stockout_param"], label="Stockout coef", color=COST_COLORS["stockout"], linewidth=2, where='post')
    ax.step(x_axis, baseline_data["ordering_param"], label="Ordering coef", color=COST_COLORS["ordering"], linewidth=2, where='post')
    ax.set_title(f"{period_name}: Cost Parameter Multipliers (Baseline view)", fontweight='bold')
    ax.set_xlabel("Day")
    ax.set_ylabel("Coefficient value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.step(x_axis, rl_data["holding_param"],  label="Holding coef",  color=COST_COLORS["holding"],  linewidth=2, where='post')
    ax.step(x_axis, rl_data["stockout_param"], label="Stockout coef", color=COST_COLORS["stockout"], linewidth=2, where='post')
    ax.step(x_axis, rl_data["ordering_param"], label="Ordering coef", color=COST_COLORS["ordering"], linewidth=2, where='post')
    ax.set_title(f"{period_name}: Cost Parameter Multipliers (RL view)", fontweight='bold')
    ax.set_xlabel("Day")
    ax.set_ylabel("Coefficient value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / output_name
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path

def create_results_directory():
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, timestamp

def save_metrics_json(
    results_dir,
    train_baseline_metrics,
    test_baseline_metrics,
    train_metrics,
    test_metrics,
    training_timesteps,
    train_days,
    test_days,
):
    """Save metrics as JSON with explicit train/test separation."""
    test_cost_reduction = test_baseline_metrics['total_cost'] - test_metrics['total_cost']
    test_cost_reduction_pct = (test_cost_reduction / test_baseline_metrics['total_cost']) * 100
    data = {
        "timestamp": datetime.now().isoformat(),
        "periods": {
            "training_timesteps": int(training_timesteps),
            "train_days": int(train_days),
            "test_days": int(test_days),
        },
        "train": {
            "baseline": {
                "total_cost": float(train_baseline_metrics['total_cost']),
                "service_level": float(train_baseline_metrics['service_level']),
                "avg_inventory": float(train_baseline_metrics['avg_inventory']),
                "stockout_events": float(train_baseline_metrics['stockout_events']),
            },
            "rl": {
                "total_cost": float(train_metrics['total_cost']),
                "service_level": float(train_metrics['service_level']),
                "avg_inventory": float(train_metrics['avg_inventory']),
                "stockout_events": float(train_metrics['stockout_events']),
            },
        },
        "test": {
            "baseline": {
                "total_cost": float(test_baseline_metrics['total_cost']),
                "service_level": float(test_baseline_metrics['service_level']),
                "avg_inventory": float(test_baseline_metrics['avg_inventory']),
                "stockout_events": float(test_baseline_metrics['stockout_events']),
            },
            "rl": {
                "total_cost": float(test_metrics['total_cost']),
                "service_level": float(test_metrics['service_level']),
                "avg_inventory": float(test_metrics['avg_inventory']),
                "stockout_events": float(test_metrics['stockout_events']),
            },
        },
        "improvement": {
            "test_cost_reduction_pct": float(test_cost_reduction_pct),
            "test_cost_saved": float(test_cost_reduction),
            "test_service_improvement_pct": float((test_metrics['service_level'] - test_baseline_metrics['service_level']) * 100),
            "test_inventory_reduction": float(test_baseline_metrics['avg_inventory'] - test_metrics['avg_inventory']),
        }
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(data, f, indent=2)

    return data

def save_summary_text(results_dir, train_baseline_metrics, test_baseline_metrics, train_metrics, test_metrics, training_timesteps, train_days, test_days):
    """Save human-readable summary with train/test sections."""
    train_cost_reduction = train_baseline_metrics['total_cost'] - train_metrics['total_cost']
    train_cost_reduction_pct = (train_cost_reduction / train_baseline_metrics['total_cost']) * 100
    test_cost_reduction = test_baseline_metrics['total_cost'] - test_metrics['total_cost']
    test_cost_reduction_pct = (test_cost_reduction / test_baseline_metrics['total_cost']) * 100
    summary = f"""
STOCKPILOT RL TRAINING RESULTS
{'='*60}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PERIODS
{'='*60}
Training Timesteps:      {training_timesteps}
Train Days:              {train_days}
Test Days:               {test_days}

TRAIN METRICS COMPARISON
{'='*60}
{'Metric':<30} {'Baseline':<15} {'RL':<15} {'Diff':<15}
{'-'*60}
Total Cost               ${train_baseline_metrics['total_cost']:<14.2f} ${train_metrics['total_cost']:<14.2f} {train_cost_reduction_pct:+.1f}%
Service Level            {train_baseline_metrics['service_level']:<14.2%} {train_metrics['service_level']:<14.2%} {((train_metrics['service_level']-train_baseline_metrics['service_level'])*100):+.1f}%
Avg Inventory            {train_baseline_metrics['avg_inventory']:<14.1f} {train_metrics['avg_inventory']:<14.1f} {(train_metrics['avg_inventory']-train_baseline_metrics['avg_inventory']):+.1f}
Stockouts                {train_baseline_metrics['stockout_events']:<14.0f} {train_metrics['stockout_events']:<14.0f}

TEST METRICS COMPARISON
{'='*60}
{'Metric':<30} {'Baseline':<15} {'RL':<15} {'Diff':<15}
{'-'*60}
Total Cost               ${test_baseline_metrics['total_cost']:<14.2f} ${test_metrics['total_cost']:<14.2f} {test_cost_reduction_pct:+.1f}%
Service Level            {test_baseline_metrics['service_level']:<14.2%} {test_metrics['service_level']:<14.2%} {((test_metrics['service_level']-test_baseline_metrics['service_level'])*100):+.1f}%
Avg Inventory            {test_baseline_metrics['avg_inventory']:<14.1f} {test_metrics['avg_inventory']:<14.1f} {(test_metrics['avg_inventory']-test_baseline_metrics['avg_inventory']):+.1f}
Stockouts                {test_baseline_metrics['stockout_events']:<14.0f} {test_metrics['stockout_events']:<14.0f}
{'='*60}

TEST SAVINGS
{'='*60}
Cost Reduction:          {test_cost_reduction_pct:+.1f}%
Total Saved:             ${test_cost_reduction:+.2f}
Service Improvement:     {((test_metrics['service_level']-test_baseline_metrics['service_level'])*100):+.1f}%
Inventory Reduction:     {(test_baseline_metrics['avg_inventory'] - test_metrics['avg_inventory']):+.1f} units
{'='*60}
"""
    
    with open(results_dir / "summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)

def test_train_and_visualize():
    """Train RL model and visualize results with matplotlib."""
    
    # Create results directory
    results_dir, timestamp = create_results_directory()
    print(f"📁 Results will be saved to: {results_dir}")
    
    print("\n" + "=" * 60)
    print("TESTING RL TRAINING & VISUALIZATION")
    print("=" * 60)
    
    # Setup
    model_path = Path("models/test_model.zip")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    training_timesteps = 400000
    n_envs = 8
    train_days = 2000
    test_days = 365
    env_seed = 42
    env_tuning = {
        "safety_adjustment_max": 60.0,
        "coverage_penalty_coef": 0.0,
        "service_bonus": 0.0,
        "service_bonus_threshold": 0.97,
    }
    
    # Create vectorized training environment to improve sample efficiency.
    print(f"\n1️⃣  Creating training environment ({n_envs} parallel envs)...")
    train_env = make_vec_env(
        lambda: ScheduledCostSupplyChainEnv(seed=42, base_seed=42, schedule_window=30, **env_tuning),
        n_envs=n_envs,
        seed=42,
    )
    
    # Create or load model
    print("2️⃣  Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.7,
        seed=42,
        verbose=0,
    )
    
    # Train
    print(f"3️⃣  Training for {training_timesteps} timesteps...")
    model.learn(
        total_timesteps=training_timesteps,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    model.save(str(model_path))
    print("✅ Model trained and saved!")
    
    # --- Simulate a separated training-period trajectory matching the training horizon ---
    print(f"\n3.5️⃣  Simulating training-period trajectory ({train_days} days) for outputs separation...")
    train_seed = 42
    train_cost_schedule = generate_cost_schedule(train_days, seed=train_seed)
    train_baseline_metrics, train_baseline_data = run_baseline_period(
        train_seed,
        train_days,
        cost_schedule=train_cost_schedule,
        env_kwargs=env_tuning,
    )
    train_metrics, train_data = run_rl_period(
        model,
        train_seed,
        train_days,
        deterministic=True,
        cost_schedule=train_cost_schedule,
        env_kwargs=env_tuning,
    )
    print(f"✅ Train-period simulation - Cost: ${train_metrics['total_cost']:.2f}, Service: {train_metrics['service_level']:.2%}")
    
    # Evaluate test baseline
    print(f"\n4️⃣  Evaluating Baseline policy on test period ({test_days} days)...")
    test_cost_schedule = generate_cost_schedule(test_days, seed=env_seed)
    test_baseline_metrics, baseline_data = run_baseline_period(
        env_seed,
        test_days,
        cost_schedule=test_cost_schedule,
        env_kwargs=env_tuning,
    )
    print(f"✅ Test Baseline - Cost: ${test_baseline_metrics['total_cost']:.2f}, Service: {test_baseline_metrics['service_level']:.2%}")
    
    # Evaluate RL on test period
    print(f"\n5️⃣  Evaluating RL model on test period ({test_days} days)...")
    model = PPO.load(str(model_path))
    eval_seed = env_seed
    test_metrics, rl_data = run_rl_period(
        model,
        eval_seed,
        test_days,
        deterministic=True,
        cost_schedule=test_cost_schedule,
        env_kwargs=env_tuning,
    )
    print(f"✅ Test RL - Cost: ${test_metrics['total_cost']:.2f}, Service: {test_metrics['service_level']:.2%}")
    
    # Calculate test improvement
    cost_reduction = test_baseline_metrics['total_cost'] - test_metrics['total_cost']
    cost_reduction_pct = (cost_reduction / test_baseline_metrics['total_cost']) * 100
    print(f"\n💰 Cost Improvement: {cost_reduction_pct:+.1f}% (${cost_reduction:+.0f})")
    
    # Save metrics
    print(f"\n6️⃣  Saving results to {results_dir}...")
    metrics_data = save_metrics_json(
        results_dir,
        train_baseline_metrics,
        test_baseline_metrics,
        train_metrics,
        test_metrics,
        training_timesteps,
        train_days,
        test_days,
    )
    save_summary_text(
        results_dir,
        train_baseline_metrics,
        test_baseline_metrics,
        train_metrics,
        test_metrics,
        training_timesteps,
        train_days,
        test_days,
    )
    
    # Create dedicated visualizations
    print("7️⃣  Creating visualizations...")
    train_chart_path = save_period_chart(
        results_dir,
        period_name="Train",
        days=train_days,
        baseline_data=train_baseline_data,
        rl_data=train_data,
        output_name="train_period_chart.png",
    )
    print(f"✅ Train chart saved to: {train_chart_path}")

    test_chart_path = save_period_chart(
        results_dir,
        period_name="Test",
        days=test_days,
        baseline_data=baseline_data,
        rl_data=rl_data,
        output_name="test_period_chart.png",
    )
    print(f"✅ Test chart saved to: {test_chart_path}")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE!")
    print("=" * 60)
    print(f"\n📁 All results saved to: {results_dir}")
    print(f"   - metrics.json (machine-readable)")
    print(f"   - summary.txt (human-readable)")
    print(f"   - train_period_chart.png (train visualization)")
    print(f"   - test_period_chart.png (test visualization)")
    
    return {
        "train_baseline_metrics": train_baseline_metrics,
        "test_baseline_metrics": test_baseline_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_baseline_data": train_baseline_data,
        "test_baseline_data": baseline_data,
        "train_data": train_data,
        "test_data": rl_data,
        "cost_reduction_pct": cost_reduction_pct,
        "results_dir": results_dir
    }

if __name__ == "__main__":
    results = test_train_and_visualize()
