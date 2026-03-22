# StockPilot 🚀

**An AI-Powered Supply Chain Simulation with Reinforcement Learning**

StockPilot is a simulation framework that demonstrates how a Reinforcement Learning agent learns to optimize safety stock levels in a dynamic supply chain environment, with detailed training and evaluation charts saved automatically.

## Features

✨ **Core Capabilities**
- **RL-Powered Optimization**: Stable-Baselines3 PPO agent learns to balance inventory costs, stockouts, and service levels
- **Realistic Supply Chain Dynamics**: Variable lead times, stochastic demand with spikes, and multiple cost factors
- **Baseline Comparison**: Rule-based fixed safety stock policy for performance benchmarking
- **Automated Visualization**: Timestamped results directory with train/test charts and metrics saved per run
- **Cost Breakdown Analysis**: Holding, stockout, and ordering costs tracked separately for Baseline and RL

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
1. Train a PPO agent for 20 000 timesteps
2. Evaluate both Baseline and RL on train and test periods
3. Save charts, metrics JSON, and a summary text to `results/<timestamp>/`

## How It Works

### Environment State
- Current inventory level
- Safety stock level
- Incoming orders (pipeline status)
- Recent demand history
- Estimated lead time

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

Each run produces two 2×2 charts (Train and Test period):

| Position | Content |
|----------|---------|
| Top-left | Baseline Inventory vs Safety Stock line |
| Top-right | RL Inventory vs dynamic Safety Stock |
| Bottom-left | Baseline cumulative cost breakdown (Holding / Stockout / Ordering) |
| Bottom-right | RL cumulative cost breakdown (Holding / Stockout / Ordering) |

## Results & Metrics

Saved automatically to `results/<timestamp>/metrics.json`:

| Metric | Description |
|--------|-------------|
| **Total Cost** | Sum of holding + stockout + ordering costs |
| **Service Level** | % of demand fulfilled without stockouts |
| **Stockout Count** | Number of stockout events |
| **Avg Inventory** | Average inventory level over time |

---

**Status:** In Development  
**Last Updated:** March 2026
