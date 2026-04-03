"""
Metrics and analysis utilities for supply chain simulations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


class SimulationMetrics:
    """Compute and track simulation metrics."""
    
    @staticmethod
    def compute_metrics(env) -> Dict[str, float]:
        """
        Compute comprehensive metrics from environment history.
        
        Args:
            env: SupplyChainEnv instance after simulation
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if len(env.cost_history) == 0:
            return metrics
        
        # Cost metrics
        metrics["total_cost"] = float(np.sum(env.cost_history))
        metrics["avg_daily_cost"] = float(np.mean(env.cost_history))
        metrics["std_daily_cost"] = float(np.std(env.cost_history))
        
        # Inventory metrics
        metrics["avg_inventory"] = float(np.mean(env.inventory_history))
        metrics["max_inventory"] = float(np.max(env.inventory_history))
        metrics["min_inventory"] = float(np.min(env.inventory_history))
        metrics["std_inventory"] = float(np.std(env.inventory_history))
        
        # Demand and fulfillment
        total_demand = np.sum(env.demand_history)
        total_fulfilled = np.sum(env.fulfilled_demand_history)
        metrics["total_demand"] = float(total_demand)
        metrics["total_fulfilled"] = float(total_fulfilled)
        metrics["service_level"] = float(total_fulfilled / total_demand) if total_demand > 0 else 1.0
        
        # Stockout analysis
        stockout_events = sum(
            1 for d, f in zip(env.demand_history, env.fulfilled_demand_history)
            if f < d * 0.99  # More than 1% unmet
        )
        metrics["stockout_events"] = int(stockout_events)
        
        # Calculate cost components
        holding_costs = []
        stockout_costs = []
        order_costs = []
        
        for i, (demand, fulfilled) in enumerate(zip(env.demand_history, env.fulfilled_demand_history)):
            unfulfilled = demand - fulfilled
            
            # Estimate costs (approximate)
            inv = env.inventory_history[i] if i < len(env.inventory_history) else 0
            holding = inv * env.holding_cost
            stockout = unfulfilled * env.stockout_cost
            
            holding_costs.append(holding)
            stockout_costs.append(stockout)
        
        # Add ordering costs
        order_costs_total = len(env.order_history) * env.ordering_cost
        
        metrics["holding_cost_total"] = float(np.sum(holding_costs))
        metrics["stockout_cost_total"] = float(np.sum(stockout_costs))
        metrics["ordering_cost_total"] = float(order_costs_total)
        
        # Order metrics
        metrics["num_orders"] = int(len(env.order_history))
        metrics["avg_order_qty"] = float(
            np.mean([qty for _, qty, _ in env.order_history]) if env.order_history else 0
        )
        
        # Safety stock metrics
        metrics["avg_safety_stock"] = float(np.mean(env.safety_stock_history))
        
        return metrics
    
    @staticmethod
    def compare_policies(baseline_metrics: Dict, rl_metrics: Dict) -> Dict[str, Any]:
        """
        Compare baseline and RL policy metrics.
        
        Args:
            baseline_metrics: Metrics from baseline policy
            rl_metrics: Metrics from RL policy
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "baseline": baseline_metrics,
            "rl": rl_metrics,
            "improvements": {},
        }
        
        # Cost reduction
        if "total_cost" in baseline_metrics and "total_cost" in rl_metrics:
            cost_reduction = baseline_metrics["total_cost"] - rl_metrics["total_cost"]
            cost_reduction_pct = (cost_reduction / baseline_metrics["total_cost"]) * 100 if baseline_metrics["total_cost"] > 0 else 0
            comparison["improvements"]["cost_reduction"] = float(cost_reduction)
            comparison["improvements"]["cost_reduction_pct"] = float(cost_reduction_pct)
        
        # Service level improvement
        if "service_level" in baseline_metrics and "service_level" in rl_metrics:
            service_improvement = rl_metrics["service_level"] - baseline_metrics["service_level"]
            comparison["improvements"]["service_level_improvement"] = float(service_improvement)
        
        # Stockout reduction
        if "stockout_events" in baseline_metrics and "stockout_events" in rl_metrics:
            stockout_reduction = baseline_metrics["stockout_events"] - rl_metrics["stockout_events"]
            comparison["improvements"]["stockout_reduction"] = int(stockout_reduction)
        
        return comparison


def create_simulation_dataframe(env) -> pd.DataFrame:
    """
    Create pandas DataFrame from simulation history.
    
    Args:
        env: SupplyChainEnv instance
        
    Returns:
        DataFrame with all timestep data
    """
    data = {
        "step": np.arange(len(env.cost_history)),
        "demand": env.demand_history,
        "fulfilled_demand": env.fulfilled_demand_history,
        "inventory": env.inventory_history,
        "safety_stock": env.safety_stock_history,
        "daily_cost": env.cost_history,
        "lead_time": env.lead_time_history,
    }
    
    return pd.DataFrame(data)


def get_cost_breakdown(env) -> Dict[str, float]:
    """
    Break down total cost by component.
    
    Args:
        env: SupplyChainEnv instance
        
    Returns:
        Dictionary with cost breakdown
    """
    metrics = SimulationMetrics.compute_metrics(env)
    
    total = metrics.get("total_cost", 1)  # Avoid division by zero
    
    return {
        "holding_cost_pct": (metrics.get("holding_cost_total", 0) / total) * 100,
        "stockout_cost_pct": (metrics.get("stockout_cost_total", 0) / total) * 100,
        "ordering_cost_pct": (metrics.get("ordering_cost_total", 0) / total) * 100,
    }
