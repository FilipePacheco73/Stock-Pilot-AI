# StockPilot 🚀

**An AI-Powered Supply Chain Simulation with Reinforcement Learning**

StockPilot is an interactive visual simulation that demonstrates how a Reinforcement Learning agent learns to optimize safety stock levels in a dynamic supply chain environment.

## Features

✨ **Core Capabilities**
- **RL-Powered Optimization**: Stable-Baselines3 agent learns to balance inventory costs, stockouts, and service levels
- **Realistic Supply Chain Dynamics**: Variable lead times, stochastic demand with spikes, and multiple cost factors
- **Interactive Dashboard**: Real-time Streamlit visualization with metrics, charts, and comparison analysis
- **Baseline Comparison**: Rule-based fixed safety stock policy for performance benchmarking
- **Professional Visualization**: Color-coded inventory status, animated timelines, and comprehensive metrics

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
│   ├── simulation/           # Core simulation logic
│   │   └── simulator.py
│   └── utils/
│       └── metrics.py
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── models/                  # Trained models & logs
├── data/                    # Simulation results
└── requirements.txt
```

## Tech Stack

- **Simulation**: Custom Gym-style environment
- **RL Framework**: Stable-Baselines3 (PPO agent)
- **Visualization**: Streamlit dashboard
- **Data Processing**: NumPy, Pandas
- **Charting**: Plotly

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train the RL Agent

```bash
python src/agents/rl_trainer.py
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## How It Works

### Environment State
- Current inventory level
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

## Results & Metrics

The dashboard displays:

| Metric | Description |
|--------|-------------|
| **Total Cost** | Sum of holding + stockout + ordering costs |
| **Service Level** | % of demand fulfilled without stockouts |
| **Stockout Count** | Number of stockout events |
| **Avg Inventory** | Average inventory level over time |

**RL Agent vs Baseline:**
- Typically achieves 15-25% cost reduction
- Higher service level with smarter inventory management
- Adaptive response to demand variability

## Dashboard Features

📊 **Visualizations:**
- Inventory levels over time (with safety stock overlay)
- Orders placed and delivery tracking
- Demand vs fulfilled demand
- Cost breakdown analysis
- Side-by-side RL vs Baseline comparison

🎨 **Color Coding:**
- 🟢 Green: Healthy inventory levels
- 🔴 Red: Stockout events
- 🟡 Yellow: Warning zone

## LinkedIn Content

Perfect for engagement! The project demonstrates:
- Practical AI applications in supply chain
- RL solving real business problems
- Visual storytelling of AI decision-making
- Quantifiable cost savings

---

**Status:** In Development  
**Last Updated:** March 2026
