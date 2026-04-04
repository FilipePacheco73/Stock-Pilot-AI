# StockPilot

StockPilot is a supply-chain simulation project built around a custom Gymnasium-style environment and a PPO agent from Stable-Baselines3. The current setup compares a rule-based baseline with an RL policy that dynamically adjusts safety stock under stochastic demand, variable lead times, and changing cost regimes.

The project currently has two main user-facing flows:
- `test_visualization.py`: train, evaluate, and export timestamped benchmark artifacts
- `dashboard/app.py`: interactive Streamlit dashboard with live Manual vs RL comparison

## Current Project State

The current benchmark and dashboard are aligned around a cost-first setup:
- PPO training with `400000` timesteps
- `8` parallel environments during training
- cost schedule randomization every `30` days
- `safety_adjustment_max = 60.0`
- `coverage_penalty_coef = 0.0`
- `service_bonus = 0.0`

In other words, the current comparison is tuned to minimize total cost, not to preserve service level.

## Latest Benchmark

Latest benchmark artifacts were generated in `results/2026-04-04_08-50-36/`.

Test-period result versus the baseline with safety stock `300`:
- Baseline total cost: `$50,103.80`
- RL total cost: `$47,977.70`
- RL cost reduction: `+4.2%`
- RL saved: `$2,126.10`

Files produced by a benchmark run:
- `metrics.json`
- `summary.txt`
- `train_period_chart.png`
- `test_period_chart.png`

## Project Structure

```text
Stock-Pilot-AI/
├── dashboard/
│   └── app.py
├── models/
│   └── checkpoints/
├── results/
│   └── YYYY-MM-DD_HH-MM-SS/
├── src/
│   ├── agents/
│   │   ├── baseline_policy.py
│   │   └── rl_trainer.py
│   ├── environment/
│   │   ├── demand_profiles.py
│   │   └── supply_chain_env.py
│   └── utils/
│       └── metrics.py
├── tests/
│   └── test_quick.py
├── main.py
├── run_tests.py
├── test_visualization.py
└── requirements.txt
```

## Environment Model

The environment models a single-stage inventory control problem with:
- stochastic daily demand with occasional spikes
- variable lead times from `1` to `5` days
- holding, stockout, and ordering costs
- automatic replenishment logic based on inventory position and safety stock

### Observation State

The observation is 8-dimensional:
- current inventory
- current safety stock
- pipeline quantity
- 7-day average demand
- estimated lead time
- current holding cost
- current stockout cost
- current ordering cost

### Action

The RL action is a continuous safety-stock adjustment signal.

The environment converts the normalized PPO action into a real safety-stock adjustment bounded by `safety_adjustment_max`.

### Reward

The base environment supports cost shaping, but the current benchmark and dashboard configuration use pure cost-focused training:

```text
reward = -(holding_cost + stockout_cost + ordering_cost)
```

That is achieved in the active benchmark/dashboard tuning by setting:
- `coverage_penalty_coef = 0.0`
- `service_bonus = 0.0`

## Main Components

### `test_visualization.py`

This is the main benchmark script.

It does the following:
- trains a PPO model with domain-randomized costs
- evaluates baseline and RL on a train period and a test period
- writes timestamped metrics and a text summary
- exports charts comparing inventory behavior and cumulative costs

### `dashboard/app.py`

The dashboard runs a live comparison between:
- a manual scenario where the user sets target safety stock
- an RL scenario that adapts safety stock automatically

On startup, the dashboard trains or refreshes the RL model before starting the live simulation.

### `main.py`

This is the older pipeline entry point for generic environment testing, baseline evaluation, and PPO training through `RLTrainer`.

It still works, but the most up-to-date benchmark path is `test_visualization.py`.

## How To Run

### 1. Install dependencies

If you are using the project virtual environment in this repository on Windows:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Generic command:

```bash
pip install -r requirements.txt
```

### 2. Run the benchmark

Windows using the existing project environment:

```powershell
.\venv\Scripts\python.exe test_visualization.py
```

Generic command:

```bash
python test_visualization.py
```

This run will:
- train PPO with the current cost-first configuration
- compare RL against the fixed baseline
- save timestamped outputs under `results/`

### 3. Run the dashboard

Windows using the existing project environment:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Generic command:

```bash
streamlit run dashboard/app.py
```

What to expect:
- the app retrains or refreshes the RL policy at startup
- the page then shows rolling Manual vs RL comparison over the last 365 simulated days
- sidebar controls let you change cost multipliers and manual safety stock

### 4. Run the quick tests

Windows using the existing project environment:

```powershell
.\venv\Scripts\python.exe run_tests.py
```

Generic command:

```bash
python run_tests.py
```

### 5. Run the older training pipeline

```bash
python main.py
```

Use this if you want the generic trainer flow in `src/agents/rl_trainer.py`. For the latest benchmark logic and exported charts, prefer `test_visualization.py`.

## Output Artifacts

Each benchmark run writes a new timestamped directory under `results/` containing:
- `metrics.json`: machine-readable metrics for train/test periods
- `summary.txt`: human-readable comparison summary
- `train_period_chart.png`: train-period chart set
- `test_period_chart.png`: test-period chart set

The charts include:
- baseline inventory trajectory
- RL inventory and adaptive safety stock trajectory
- cumulative cost breakdown for baseline and RL
- cost parameter schedules over time

## Dependencies

Core dependencies used by the current project:
- `gymnasium`
- `stable-baselines3`
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `streamlit`
- `plotly`

All are listed in `requirements.txt`.

## Notes

- The repository currently uses `venv/` as the project virtual environment directory.
- The latest tuned benchmark path is `test_visualization.py`.
- The dashboard has been aligned to the same pure-cost training pattern used by the latest benchmark.

## Version

- Current documented release: `0.5.1`
- Last updated: April 2026
