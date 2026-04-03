# StockPilot 🚀

**An AI-Powered Supply Chain Simulation with Reinforcement Learning**

StockPilot is a simulation framework that demonstrates how a Reinforcement Learning agent learns to optimize safety stock levels in a dynamic supply chain environment, with detailed training and evaluation charts saved automatically.

## Features

✨ **Core Capabilities**
- **RL-Powered Optimization**: Stable-Baselines3 PPO agent learns to balance inventory costs, stockouts, and service levels
- **Realistic Supply Chain Dynamics**: Variable lead times, stochastic demand with spikes, and multiple cost factors
- **Baseline Comparison**: Rule-based fixed safety stock policy for performance benchmarking
- **Automated Visualization**: 3×2 timestamped charts showing inventory, safety stock evolution, and cost multiplier parameters
- **Cost Breakdown Analysis**: Holding, stockout, and ordering costs tracked separately for Baseline and RL
- **Domain Randomization**: Cost schedules change every 30 days during training, creating adaptive optimization challenges
- **Contextual Agent**: Observation space includes cost multipliers, enabling context-aware decision making

## Project Structure

```
stock-pilot-ai/
├── src/
│   ├── environment/          # Custom Gym environment
│   │   ├── supply_chain_env.py
│   │   └── demand_profiles.py
│   ├── agents/               # Baseline and RL policies
│   │   ├── baseline_policy.py
│   │   └── rl_trainer.py
│   └── utils/
│       └── metrics.py
├── results/                  # Timestamped run outputs
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── train_period_chart.png
│       ├── test_period_chart.png
│       ├── metrics.json
│       └── summary.txt
├── models/                   # Trained model checkpoints
├── tests/                    # Test suite
├── test_visualization.py     # Main training & evaluation script
└── requirements.txt
```

## Tech Stack

- **Simulation**: Custom Gym-style environment
- **RL Framework**: Stable-Baselines3 (PPO agent)
- **Visualization**: Matplotlib (PNG charts per run)
- **Data Processing**: NumPy, Pandas

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train the RL Agent & Generate Charts

```bash
python test_visualization.py
```

This will:
1. Train a PPO agent for 100,000 timesteps with domain randomization (cost schedule changes every 30 days)
2. Evaluate both Baseline and RL on train and test periods with synchronized cost schedules
3. Save 3×2 charts (inventory, safety stock, cost multiplier evolution), metrics JSON, and summary to `results/<timestamp>/`
4. Display performance comparison: typically **+3-8% cost reduction** vs Baseline

## How It Works

### Environment State (8-Dimensional)
- Current inventory level
- Safety stock level
- Incoming orders (pipeline status)
- Recent demand history (7-day average)
- Estimated lead time
- **Current holding cost multiplier** (domain randomization signal)
- **Current stockout cost multiplier** (domain randomization signal)
- **Current ordering cost multiplier** (domain randomization signal)

### RL Agent Actions
- Adjust safety stock level dynamically
- Place replenishment orders

### Reward Function
Maximize efficiency by balancing:
- ✅ Minimizing stockouts (lost sales)
- ✅ Minimizing excess inventory (holding costs)
- ✅ Minimizing total operational cost

## Scenario

**Supply Chain Structure:**
- 1 Supplier with variable lead times
- 1 Factory producing final products
- Customers with uncertain demand
- Discrete time-step simulation (days)

**Key Dynamics:**
- Random demand with variability and spikes
- Variable lead times (1-5 days)
- Holding costs, stockout penalties, ordering costs

## Output Charts

Each run produces two 3×2 charts (Train and Test period):

| Position | Content |
|----------|----------|
| **Row 1** | |
| Left | Baseline Inventory vs Safety Stock |
| Right | RL Inventory vs Adaptive Safety Stock (oscillating in response to cost changes) |
| **Row 2** | |
| Left | Baseline cumulative cost breakdown (Holding / Stockout / Ordering) with total |
| Right | RL cumulative cost breakdown (Holding / Stockout / Ordering) with total |
| **Row 3** | |
| Left | Holding cost multiplier evolution (30-day step changes) |
| Right | Stockout & Ordering cost multipliers (step-function visualization) |

## Results & Metrics

Saved automatically to `results/<timestamp>/metrics.json`:

| Metric | Description |
|--------|-------------|
| **Total Cost** | Sum of holding + stockout + ordering costs |
| **Service Level** | % of demand fulfilled without stockouts |
| **Stockout Count** | Number of stockout events |
| **Avg Inventory** | Average inventory level over time |

## Latest Results (v0.4.0)

**Test Period Performance:**
- **Cost Reduction**: +3.8% RL vs Baseline ($48,187 saved $1,916)
- **Service Level**: 93.91% (Baseline 97.01%) — intentional trade-off for cost optimization
- **Inventory**: 161.1 units avg (Baseline 209.2) — 22.9% reduction
- **Stockout Events**: 30 (Baseline 12) — acceptable for cost-focused strategy

**Training Setup:**
- 100,000 timesteps with domain randomization
- Action scaling: safety_stock_adj ∈ [-40, +40] units/step
- Service bonus: 0.0 (cost-first optimization)
- Coverage penalty: 0.1 (maintains minimum service floor)

---

**Status:** Stable  
**Last Updated:** April 2026  
**Version:** 0.4.0
