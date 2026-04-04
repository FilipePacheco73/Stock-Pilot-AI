# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.5.1] - 2026-04-04

### Changed

**Dashboard RL Alignment**
- Aligned `dashboard/app.py` training setup with the current pure-cost tuning used by `test_visualization.py`
- Increased dashboard retraining target from `200,000` to `400,000` timesteps
- Switched dashboard PPO collection to `8` parallel environments for the same higher-throughput training pattern used in the visualization benchmark
- Increased dashboard `safety_adjustment_max` from `40.0` to `60.0`
- Removed dashboard reward shaping by setting `coverage_penalty_coef=0.0` and keeping `service_bonus=0.0`
- Updated fresh dashboard PPO configuration to use `batch_size=256` and `vf_coef=0.7`

### Validated

**Latest Pure-Cost Benchmark**
- Latest `test_visualization.py` run achieved `+4.2%` test cost reduction versus the baseline (`$47,977.70` vs `$50,103.80`)
- Benchmark artifacts saved under `results/2026-04-04_08-50-36/`

## [0.5.0] - 2026-04-03

### Added

**Live Streamlit Dashboard**
- Restored `dashboard/app.py` as an interactive end-user interface for continuous supply-chain simulation
- Added side-by-side Manual vs RL scenario monitoring with rolling 365-day window updates
- Added real-time user controls for holding, stockout, and ordering cost multipliers
- Added manual-control scenario where the user directly sets the target safety stock
- Added live RL training progress feedback during app initialization

**Continuous Scenario Playback**
- Added `ReplayScenarioEnv` to replay the same demand and lead-time schedule in both Manual and RL scenarios for fair comparison
- Added rolling comparison of the last 365 days between Manual and RL policies
- Added cost chart based on 365-day moving average of daily cost

**Dashboard Training Domain Randomization**
- Added `ScheduledCostTrainingEnv` for PPO training inside the dashboard with dynamic cost multipliers during learning
- Added `generate_cost_multiplier_schedule()` to vary holding, stockout, and ordering multipliers in fixed windows during training
- Dashboard training now exposes cost multipliers in the observation state and retrains on cost-varying scenarios before live simulation

### Changed

**Dashboard RL Training**
- Dashboard initialization now always retrains the RL policy on startup instead of relying only on a pre-trained checkpoint
- Training target in the dashboard increased to `200,000` timesteps for stronger adaptation to dynamic cost scenarios
- Compatible checkpoints can still be loaded as a starting point, but they are retrained during initialization

**Safety Stock Range**
- Expanded manual safety-stock control from `300` to `500`
- Updated environment support so both Manual and RL policies can operate with safety stock up to `500`
- Made safety stock bounds configurable in `SupplyChainEnv` via `safety_stock_min` and `safety_stock_max`

**Dashboard Visualization**
- Simplified upper charts to show only inventory and safety stock, one panel for Manual and one for RL
- Moved cost comparison into a dedicated chart below the scenario panels
- Replaced deprecated Streamlit `use_container_width` usage with `width="stretch"`

### Fixed

**Dashboard Decision Context**
- Ensured the RL policy receives observations updated with the current cost multipliers before each live decision
- Aligned the training environment so the next observation reflects the scheduled costs consistently across steps

**Environment Warnings**
- Fixed Gymnasium `Box` precision warning by defining `low` and `high` arrays explicitly as `float32`

---

## [0.4.0] - 2026-04-03

### Added

**Domain Randomization & Dynamic Cost Scenarios**
- `ScheduledCostSupplyChainEnv` wrapper for applying cost schedule per episode (30-day regime changes)
- `generate_cost_schedule()` function to pre-generate cost multiplier sequences for reproducible randomization
- Cost multipliers now part of observation state (8-dimensional: inventory, safety_stock, pipeline_qty, avg_demand_7d, lead_time_est, holding_cost, stockout_cost, ordering_cost)
- Synchronized cost schedules between Baseline and RL for fair evaluation

**Enhanced Visualization**
- Upgraded from 2×2 to 3×2 grid layout for train/test period charts
- Row 3: Step-function plots showing cost multiplier evolution over 30-day windows (Holding, Stockout, Ordering)
- Added total cost to legend in cumulative cost breakdown panels
- Visual confirmation of domain randomization effects

### Changed

**Cost Formula Refinement**
- Holding cost now applies to total inventory level instead of only excess above safety stock
- Updated metric calculation in `src/utils/metrics.py` to align with environment implementation
- More realistic total-inventory-based cost accounting

**RL Agent Hyperparameter Tuning**
- Increased training timesteps: 20,000 → 100,000 for improved convergence with domain randomization
- Increased action scaling: 10.0 → 40.0 units per step (safety_adjustment_max) for adaptive safety stock
- Reduced service bonus: 5.0 → 0.0 to prioritize cost minimization over service level maintenance
- Coverage penalty coefficient: 0.1 (maintained for service floor)
- New tuning parameters in `env_tuning` dict passed to all evaluation runs

**Seed Management**
- Aligned random seed between Baseline and RL evaluation on same test period for fair comparison
- Baseline initialized with fixed safety_stock=300 and reorder_point=300 (from v0.3.0)

### Validated

**Performance Improvement**
- **Test Period**: RL achieved +3.8% cost reduction vs Baseline ($48,187 vs $50,104, saved $1,916)
- **Service Trade-off**: Cost optimization intentionally reduces service level from 97.01% → 93.91% (3% acceptable trade-off)
- **Inventory Reduction**: Average inventory decreased from 209.2 → 161.1 units (-48.2 units, -22.9%)
- **Stockout Events**: Increased from 12 → 30 (expected due to cost minimization focus)
- **Train Period**: +8.6% cost reduction on training data (indicating good generalization potential)

---

## [0.3.0] - 2026-03-22

### Removed

**Dashboard**
- Deleted `dashboard/` directory (`app.py` and all related Streamlit code) — to be redesigned in a future release
- Removed Streamlit and Plotly as active dependencies for the main workflow

### Changed

**Visualization (`test_visualization.py`)**
- Unified color palette across all 4 chart panels (Train and Test): blue = inventory/holding, red = safety stock/stockout, green = ordering
- Split cumulative cost chart into two separate breakdown panels: Baseline (bottom-left) and RL (bottom-right)
- Each cost breakdown panel shows Holding, Stockout, and Ordering as individual lines with consistent colors
- Removed the combined "Cumulative Cost Comparison" panel (Baseline + RL overlapped)
- Removed the `colors` parameter from `save_period_chart()` — palette is now hardcoded inside the function for consistency across Train and Test runs
- Baseline policy initialised with `safety_stock=300` and `reorder_point=300` (previously 75) for a fairer comparison

### Added

**Cost tracking per step**
- `run_baseline_period()` and `run_rl_period()` now collect `holding_costs`, `stockout_costs`, and `ordering_costs` individually per day (previously only the total was stored)

---

## [0.2.0] - 2026-03-22

### Changed

**Dashboard Redesign - Complete Refactor**
- Rebuilt dashboard from scratch with simplified, clean interface
- Moved training controls to main area (not sidebar)
- Single training button in main content for 1-period training
- Removed multi-period training selector (users train incrementally)
- Sidebar now contains ONLY configuration parameters
- Much cleaner user experience with focus on key actions

**Dashboard Architecture**
- Reduced file complexity by removing complex state management
- Simplified plotting functions (removed evolution charts)
- Session state now minimal: training_done, simulation_results
- Focus on immediate results rather than training history
- Cleaner main() function structure
- Removed DASHBOARD_GUIDE.md concepts - now very intuitive

**Interface Layout**
- Header: Title and description
- Sidebar: Configuration options (seed, demand, cost parameters)
- Main area: Three action buttons (Train, Status, Simulate)
- Results: Metrics comparison + Improvement KPIs + Visualizations
- All in one coherent flow

### Removed

**Overly Complex Features**
- Multi-period training selection (1/5/10 periods)
- Training evolution 4-metric subplot visualization
- Training history tracking and display
- "Show Training Results" button
- Checkpoint management UI
- Model status in sidebar

---

## [0.1.1] - 2026-03-22

### Improved

**Data Type Refinement**
- Integer-only demand values for clarity and realism
- Integer inventory levels throughout simulation
- Integer fulfilled demand quantities
- Rounded stochastic values to nearest integer using `int(np.round())`
- Better visualization in dashboards and reports

**Codebase Quality**
- Type hints for demand generation methods
- Consistent integer handling in pipeline operations
- Improved order quantity conversion to integers
- Better variable naming for clarity
- Enhanced code documentation

**Project Organization**
- Created dedicated `tests/` folder for test suite organization
- Moved test files into proper test module structure
- Created master test runner (`run_tests.py`) for unified execution
- Added `__init__.py` to make tests a proper Python package
- Removed redundant test files from project root

**Testing Infrastructure**
- Centralized test execution via `run_tests.py`
- Clean test output with organized reporting
- Quick validation of environment and policies
- Integer value validation tests
- Test discovery and execution framework

### Changed

**Core Environment**
- `_generate_demand()` now returns `int` instead of `float`
- `_process_pipeline()` now returns `int` instead of `float`
- Action processing rounds order quantities to integers
- All inventory operations use integer arithmetic
- Order quantity threshold changed from `0.1` to `0`

**Project Structure**
- Tests moved from root to dedicated `tests/` directory
- Test runner refactored as master entry point
- Cleaner project root with only essential files
- Better separation of concerns

**Documentation**
- Added PROJECT_REVIEW.md for project status overview
- Enhanced CHANGELOG with detailed improvement tracking
- Updated project organization documentation
- Added refinement log for tracking iterative improvements

### Validation

**Tests Passing**
- Environment creation and initialization
- Demand generation (all values are integers)
- Inventory tracking (all values are integers)
- Baseline policy evaluation (365-day simulation)
- Random policy comparison
- All imports and module structure

**Performance Verified**
- Baseline Policy: $76,415.50 cost, 63.89% service level
- Integer conversion maintains numerical accuracy
- No data type mismatches or rounding errors
- Clean separation between float actions and integer states

## [0.1.0] - 2026-03-22

### Added

**Project Foundation**
- Initial repository setup with core structure
- Custom Gym-style supply chain environment
- Support for stochastic demand with variability and spikes
- Variable lead time dynamics (1-5 days uncertainty)
- Multi-cost system: holding costs, stockout penalties, ordering costs

**RL Agent Integration**
- Stable-Baselines3 PPO agent implementation
- Agent training pipeline with configurable hyperparameters
- State representation: inventory, pipeline, demand history, lead time
- Action space: safety stock adjustment and reorder decisions
- Reward function optimized for cost minimization and service level

**Baseline Policy**
- Fixed safety stock rule-based policy
- Deterministic reorder strategy for comparison
- Performance metrics baseline for RL evaluation

**Simulation Engine**
- Discrete time-step simulation (days)
- Order fulfillment and delivery pipeline
- Demand generation with randomness
- Inventory state tracking and cost calculation
- Metrics collection (service level, stockouts, costs)

**Dashboard & Visualization**
- Streamlit interactive dashboard
- Real-time inventory level charts
- Safety stock tracking visualization
- Order and delivery timeline
- Demand vs fulfilled demand comparison
- RL agent vs Baseline policy side-by-side analysis
- Color-coded status indicators (green/red/yellow)
- Comprehensive metrics display

**Documentation**
- Comprehensive README with quick-start guide
- Project structure documentation
- Tech stack specification
- Feature overview and capabilities
- Dashboard features documentation

### Technical Specifications

**Core Components**
- Environment: Custom environment with realistic supply chain dynamics
- RL Framework: Stable-Baselines3 with PPO algorithm
- Visualization: Streamlit for interactive dashboards
- Data Pipeline: NumPy/Pandas for metrics and analysis

**Performance Targets**
- Expected 15-25% cost reduction over baseline policy
- High service level (>95% demand fulfillment)
- Real-time dashboard updates
- Scalable to longer time horizons (1000+ days)

**Key Features**
- Adaptive safety stock levels based on learned policy
- Intelligent response to demand variability
- Balanced optimization across multiple objectives
- Visual feedback on system performance
- Easy-to-understand metrics for non-technical users