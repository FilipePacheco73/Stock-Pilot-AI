# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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