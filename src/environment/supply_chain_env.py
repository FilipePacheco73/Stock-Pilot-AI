"""
Supply Chain Environment for Reinforcement Learning
Simulates a factory with supplier inventory management
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any


class SupplyChainEnv(gym.Env):
    """
    Supply Chain Environment with variable demand and lead times.
    
    State:
    - Current inventory level
    - Pipeline (orders in transit)
    - Recent demand history
    - Estimated lead time
    
    Action:
    - Safety stock adjustment (continuous, normalized)
    - MRP computes order quantity from shortage automatically
    
    Reward:
    - Negative cost: holding + stockout + ordering
    - Positive bonus for service level
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        initial_inventory: int = 100,
        holding_cost: float = 0.5,
        stockout_cost: float = 10.0,
        ordering_cost: float = 5.0,
        lead_time_min: int = 1,
        lead_time_max: int = 5,
        demand_mean: float = 50.0,
        demand_std: float = 15.0,
        demand_spike_prob: float = 0.1,
        demand_spike_multiplier: float = 2.5,
        episode_length: int = 365,
        max_order_qty: int = 500,
        initial_safety_stock: int = 75,
        seed: int = None,
    ):
        """
        Initialize the supply chain environment.
        
        Args:
            initial_inventory: Starting inventory level
            holding_cost: Cost per unit held in inventory per day
            stockout_cost: Cost per unit of unmet demand
            ordering_cost: Fixed cost per order
            lead_time_min: Minimum lead time in days
            lead_time_max: Maximum lead time in days
            demand_mean: Mean daily demand
            demand_std: Standard deviation of demand
            demand_spike_prob: Probability of demand spike
            demand_spike_multiplier: Multiplier for spike demand
            episode_length: Number of days per episode
            max_order_qty: Maximum replenishment quantity per step
            initial_safety_stock: Initial safety stock level
            seed: Random seed
        """
        super().__init__()
        
        # Environment parameters
        self.initial_inventory = initial_inventory
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost
        self.lead_time_min = lead_time_min
        self.lead_time_max = lead_time_max
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.demand_spike_prob = demand_spike_prob
        self.demand_spike_multiplier = demand_spike_multiplier
        self.episode_length = episode_length
        self.max_order_qty = max_order_qty
        self.initial_safety_stock = initial_safety_stock
        
        # RNG
        self.rng = np.random.RandomState(seed)
        
        # State variables
        self.inventory = initial_inventory
        self.safety_stock = int(np.clip(initial_safety_stock, 5, 300))
        self.pipeline = []  # List of (quantity, days_remaining)
        self.current_step = 0
        
        # History
        self.demand_history = []
        self.inventory_history = []
        self.safety_stock_history = []
        self.order_history = []
        self.cost_history = []
        self.lead_time_history = []
        self.fulfilled_demand_history = []
        
        # Agent chooses only safety-stock adjustment (normalized).
        # action[0] in [-1, 1] -> safety-stock adjustment in [-10, +10] units
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # Observation space: [inventory, safety_stock, pipeline_qty, avg_demand_7d, lead_time_est]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            high=np.array([500.0, 300.0, 500.0, 200.0, 5.0]),
            dtype=np.float32
        )
        
    def _generate_demand(self) -> int:
        """Generate stochastic demand with occasional spikes (returns integer)."""
        base_demand = self.rng.normal(self.demand_mean, self.demand_std)
        base_demand = max(0, base_demand)  # Demand can't be negative
        
        # Demand spike
        if self.rng.random() < self.demand_spike_prob:
            base_demand *= self.demand_spike_multiplier
            
        return int(np.round(base_demand))
    
    def _generate_lead_time(self) -> int:
        """Generate variable lead time."""
        return self.rng.randint(self.lead_time_min, self.lead_time_max + 1)
    
    def _get_state(self) -> np.ndarray:
        """Get current observation state."""
        # Recent demand average (7 days or less if not enough history)
        if len(self.demand_history) > 0:
            avg_demand_7d = np.mean(self.demand_history[-7:])
        else:
            avg_demand_7d = self.demand_mean
        
        # Total pipeline quantity
        pipeline_qty = sum(qty for qty, _ in self.pipeline)
        
        # Estimated lead time (average of pipeline)
        if len(self.pipeline) > 0:
            lead_time_est = np.mean([days for _, days in self.pipeline])
        else:
            lead_time_est = (self.lead_time_min + self.lead_time_max) / 2
        
        state = np.array([
            float(self.inventory),
            float(self.safety_stock),
            float(pipeline_qty),
            float(avg_demand_7d),
            float(lead_time_est)
        ], dtype=np.float32)
        
        return state

    def _decode_action(self, action: np.ndarray) -> Tuple[float, float]:
        """Map normalized PPO action into safety-stock adjustment."""
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        safety_signal = float(clipped_action[0])
        safety_stock_adj = safety_signal * 10.0
        return safety_stock_adj, safety_signal

    def encode_action(self, safety_stock_adj: float = 0.0, *_unused) -> np.ndarray:
        """Map safety-stock adjustment into normalized action space."""
        safety_signal = np.clip(float(safety_stock_adj) / 10.0, -1.0, 1.0)
        return np.array([safety_signal], dtype=np.float32)
    
    def _process_pipeline(self) -> int:
        """Process deliveries from pipeline (returns integer)."""
        delivered = 0
        new_pipeline = []
        
        for qty, days_remaining in self.pipeline:
            if days_remaining <= 1:
                delivered += int(qty)
            else:
                new_pipeline.append((int(qty), days_remaining - 1))
        
        self.pipeline = new_pipeline
        return delivered
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of environment.
        
        Args:
            action: [safety_stock_adjustment_signal]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Parse normalized action into safety-stock adjustment.
        safety_stock_adj, safety_signal = self._decode_action(action)
        self.safety_stock = int(np.clip(self.safety_stock + safety_stock_adj, 5, 300))
        
        # Process deliveries from pipeline
        delivered = self._process_pipeline()
        self.inventory += delivered
        
        # MRP ordering: order according to shortage in inventory position.
        ordering_cost_incurred = 0.0
        avg_demand = np.mean(self.demand_history[-7:]) if self.demand_history else self.demand_mean
        lead_time_est = np.mean([days for _, days in self.pipeline]) if self.pipeline else (self.lead_time_min + self.lead_time_max) / 2
        target_inventory_position = self.safety_stock + (avg_demand * lead_time_est)
        inventory_position = self.inventory + sum(qty for qty, _ in self.pipeline)
        shortage = max(0.0, target_inventory_position - inventory_position)
        order_qty = int(np.clip(np.round(shortage), 0, self.max_order_qty))

        if order_qty > 0:  # Only incur cost if order is significant
            lead_time = self._generate_lead_time()
            self.pipeline.append((order_qty, lead_time))
            ordering_cost_incurred = self.ordering_cost
            self.order_history.append((self.current_step, order_qty, lead_time))
        
        # Generate demand
        demand = self._generate_demand()
        self.demand_history.append(demand)
        
        # Fulfill demand (all integer operations)
        fulfilled_demand = min(self.inventory, demand)
        unfulfilled_demand = demand - fulfilled_demand
        self.inventory -= fulfilled_demand
        self.fulfilled_demand_history.append(fulfilled_demand)
        
        # Calculate costs
        holding_cost_incurred = self.holding_cost * max(0, self.inventory - self.safety_stock)
        stockout_cost_incurred = self.stockout_cost * unfulfilled_demand
        
        total_cost = holding_cost_incurred + stockout_cost_incurred + ordering_cost_incurred

        # Dense shaping: penalize weak inventory coverage before stockouts happen.
        updated_inventory_position = self.inventory + sum(qty for qty, _ in self.pipeline)
        updated_lead_time_est = np.mean([days for _, days in self.pipeline]) if self.pipeline else (self.lead_time_min + self.lead_time_max) / 2
        updated_target_position = self.safety_stock + (avg_demand * updated_lead_time_est)
        coverage_gap = max(0.0, updated_target_position - updated_inventory_position)
        coverage_penalty = 0.2 * coverage_gap
        
        # Reward: negative cost + bonus for good service level
        service_level = fulfilled_demand / demand if demand > 0 else 1.0
        reward = -(total_cost + coverage_penalty)
        
        # Bonus for maintaining good service level
        if service_level > 0.95:
            reward += 5.0
        
        # Track history
        self.inventory_history.append(self.inventory)
        self.safety_stock_history.append(self.safety_stock)
        self.cost_history.append(total_cost)
        self.lead_time_history.append(len(self.pipeline))
        
        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Get next observation
        observation = self._get_state()
        
        # Info dict
        info = {
            "raw_action": np.asarray(action, dtype=np.float32).tolist(),
            "safety_signal": safety_signal,
            "demand": demand,
            "fulfilled_demand": fulfilled_demand,
            "unfulfilled_demand": unfulfilled_demand,
            "inventory": self.inventory,
            "safety_stock": self.safety_stock,
            "holding_cost": holding_cost_incurred,
            "stockout_cost": stockout_cost_incurred,
            "ordering_cost": ordering_cost_incurred,
            "order_qty": order_qty,
            "shortage": shortage,
            "coverage_gap": coverage_gap,
            "coverage_penalty": coverage_penalty,
            "service_level": service_level,
            "pipeline_qty": sum(qty for qty, _ in self.pipeline),
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Optional random seed
            
        Returns:
            observation, info
        """
        if seed is not None:
            self.rng.seed(seed)
        
        self.inventory = self.initial_inventory
        self.safety_stock = int(np.clip(self.initial_safety_stock, 5, 300))
        self.pipeline = []
        self.current_step = 0
        
        # Clear history
        self.demand_history = []
        self.inventory_history = []
        self.safety_stock_history = []
        self.order_history = []
        self.cost_history = []
        self.lead_time_history = []
        self.fulfilled_demand_history = []
        
        observation = self._get_state()
        info = {}
        
        return observation, info
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of the episode."""
        if len(self.cost_history) == 0:
            return {}
        
        total_cost = sum(self.cost_history)
        avg_cost = total_cost / len(self.cost_history)
        holding_costs = [c * (self.holding_cost / (self.holding_cost + self.stockout_cost + self.ordering_cost + 0.01))
                         for c in self.cost_history]
        stockout_count = sum(1 for d, f in zip(self.demand_history, self.fulfilled_demand_history) if f < d * 0.99)
        service_level = sum(self.fulfilled_demand_history) / (sum(self.demand_history) + 0.001)
        
        return {
            "total_cost": total_cost,
            "avg_daily_cost": avg_cost,
            "holding_costs": sum(holding_costs),
            "stockouts": stockout_count,
            "service_level": service_level,
            "avg_inventory": np.mean(self.inventory_history) if self.inventory_history else 0,
        }
    
    def render(self, mode: str = "human") -> None:
        """Render the environment (placeholder)."""
        pass
