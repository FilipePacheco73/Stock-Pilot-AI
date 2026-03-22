"""
Baseline Policy: Fixed Safety Stock Strategy
"""

import numpy as np
from typing import Dict, Any


class BaselinePolicy:
    """
    Fixed safety stock baseline policy.
    
    Strategy:
    - Maintain a fixed safety stock level
    - Reorder when inventory drops below safety stock
    - Reorder quantity = mean demand * lead time + safety stock
    """
    
    def __init__(
        self,
        safety_stock: int = 75,
        reorder_point: int = 75,
        lead_time_estimate: float = 3.0,
        demand_mean: float = 50.0,
    ):
        """
        Initialize baseline policy.
        
        Args:
            safety_stock: Fixed safety stock level
            reorder_point: Inventory level to trigger reorder
            lead_time_estimate: Estimated lead time
            demand_mean: Mean daily demand for reorder calculation
        """
        self.safety_stock = safety_stock
        self.reorder_point = reorder_point
        self.lead_time_estimate = lead_time_estimate
        self.demand_mean = demand_mean
        
        # Reorder quantity = demand during lead time + safety stock
        self.reorder_qty = int(demand_mean * lead_time_estimate + safety_stock)
    
    def decide(
        self,
        inventory: float,
        pipeline_qty: float,
        demand_history: list,
        **kwargs
    ) -> Dict[str, float]:
        """
        Decide action based on current state.
        
        Args:
            inventory: Current inventory level
            pipeline_qty: Quantity in pipeline
            demand_history: Recent demand history
            
        Returns:
            Dictionary with action: {"safety_stock_adj": float, "order_qty": float}
        """
        current_safety_stock = kwargs.get("current_safety_stock", self.safety_stock)

        # Safety stock adjustment: always try to maintain target safety stock
        safety_stock_adj = self.safety_stock - current_safety_stock
        safety_stock_adj = np.clip(safety_stock_adj, -10.0, 10.0)
        
        # Reorder decision
        available_inventory = inventory + pipeline_qty
        order_qty = 0.0
        
        if available_inventory < self.reorder_point:
            order_qty = float(self.reorder_qty)
        
        return {
            "safety_stock_adj": safety_stock_adj,
            "order_qty": order_qty,
        }
    
    def get_name(self) -> str:
        """Get policy name."""
        return f"Baseline (Safety Stock={self.safety_stock})"


def run_baseline_simulation(
    env,
    policy: BaselinePolicy,
    episodes: int = 1,
    episode_length: int = 365,
) -> Dict[str, Any]:
    """
    Run simulation with baseline policy.
    
    Args:
        env: Supply chain environment
        policy: Baseline policy
        episodes: Number of episodes to run
        episode_length: Steps per episode
        
    Returns:
        Dictionary with results and metrics
    """
    all_results = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_data = {
            "demands": [],
            "inventory": [],
            "safety_stock": [],
            "costs": [],
            "fulfilled": [],
            "orders": [],
        }
        
        step = 0
        while not done and step < episode_length:
            # Get current state from observation
            inventory = obs[0]
            safety_stock = obs[1]
            pipeline_qty = obs[2]
            demand_history = env.demand_history
            
            # Get action from policy
            action_dict = policy.decide(inventory, pipeline_qty, demand_history)
            action = env.encode_action(action_dict["safety_stock_adj"])
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track data
            episode_data["demands"].append(info["demand"])
            episode_data["inventory"].append(info["inventory"])
            episode_data["safety_stock"].append(safety_stock)
            episode_data["fulfilled"].append(info["fulfilled_demand"])
            cost = info["holding_cost"] + info["stockout_cost"] + info["ordering_cost"]
            episode_data["costs"].append(cost)
            if action_dict["order_qty"] > 0:
                episode_data["orders"].append(step)
            
            done = terminated or truncated
            step += 1
        
        # Get summary
        summary = env.get_summary()
        episode_data["summary"] = summary
        all_results.append(episode_data)
    
    return {
        "policy": policy.get_name(),
        "episodes": all_results,
        "avg_total_cost": np.mean([ep["summary"]["total_cost"] for ep in all_results]),
        "avg_service_level": np.mean([ep["summary"]["service_level"] for ep in all_results]),
        "avg_stockouts": np.mean([ep["summary"]["stockouts"] for ep in all_results]),
    }
