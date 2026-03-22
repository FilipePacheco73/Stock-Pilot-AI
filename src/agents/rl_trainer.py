"""
RL Agent Training with Stable-Baselines3
"""

import numpy as np
import os
from typing import Dict, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.environment import SupplyChainEnv


class RLTrainer:
    """
    Trainer for RL agent in supply chain environment.
    Uses Stable-Baselines3 PPO algorithm.
    """
    
    def __init__(
        self,
        env_params: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None,
        model_dir: str = "models",
        log_dir: str = "logs",
        seed: int = 42,
    ):
        """
        Initialize RL trainer.
        
        Args:
            env_params: Environment configuration
            model_params: PPO model hyperparameters
            model_dir: Directory to save models
            log_dir: Directory for logs
            seed: Random seed
        """
        self.seed = seed
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Default environment parameters
        if env_params is None:
            env_params = {
                "initial_inventory": 100,
                "holding_cost": 0.5,
                "stockout_cost": 10.0,
                "ordering_cost": 5.0,
                "lead_time_min": 1,
                "lead_time_max": 5,
                "demand_mean": 50.0,
                "demand_std": 15.0,
                "demand_spike_prob": 0.1,
                "demand_spike_multiplier": 2.5,
                "episode_length": 365,
                "seed": seed,
            }
        
        self.env_params = env_params
        
        # Default model parameters
        if model_params is None:
            model_params = {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "verbose": 1,
            }
        
        self.model_params = model_params
        self.model = None
        self.train_env = None
    
    def create_env(self, n_envs: int = 1) -> "SupplyChainEnv":
        """
        Create single or vectorized environment.
        
        Args:
            n_envs: Number of parallel environments
            
        Returns:
            Environment or VecEnv
        """
        def _make_env():
            return SupplyChainEnv(**self.env_params)
        
        if n_envs == 1:
            return _make_env()
        else:
            return make_vec_env(_make_env, n_envs=n_envs, seed=self.seed)
    
    def train(
        self,
        total_timesteps: int = 500_000,
        n_envs: int = 4,
        eval_freq: int = 10_000,
        save_freq: int = 50_000,
    ) -> Tuple["PPO", Dict[str, Any]]:
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total timesteps for training
            n_envs: Number of parallel environments
            eval_freq: Evaluation frequency
            save_freq: Model saving frequency
            
        Returns:
            Trained model and training info
        """
        print(f"Creating environment with {n_envs} parallel envs...")
        self.train_env = self.create_env(n_envs=n_envs)
        
        print(f"Initializing PPO agent with hyperparameters: {self.model_params}")
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            seed=self.seed,
            **self.model_params,
            tensorboard_log=self.log_dir,
        )
        
        # Define callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self.model_dir,
            name_prefix="ppo_agent",
            save_replay_buffer=False,
        )
        
        # Evaluation environment
        eval_env = self.create_env(n_envs=1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
        )
        
        print(f"Training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, "ppo_agent_final")
        self.model.save(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        return self.model, {
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "model_path": final_model_path,
        }
    
    def evaluate(
        self,
        model_path: str = None,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate trained agent.
        
        Args:
            model_path: Path to trained model
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        # Load model if path provided
        if model_path is not None:
            self.model = PPO.load(model_path)
        
        if self.model is None:
            raise ValueError("No model loaded. Train first or provide model_path.")
        
        # Create evaluation environment
        eval_env = self.create_env(n_envs=1)
        
        all_metrics = {
            "total_costs": [],
            "service_levels": [],
            "stockouts": [],
            "avg_inventories": [],
        }
        
        for ep in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            step = 0
            
            while not done:
                if deterministic:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action, _ = self.model.predict(obs, deterministic=False)
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                step += 1
            
            # Collect metrics
            summary = eval_env.get_summary()
            all_metrics["total_costs"].append(summary.get("total_cost", 0))
            all_metrics["service_levels"].append(summary.get("service_level", 0))
            all_metrics["stockouts"].append(summary.get("stockouts", 0))
            all_metrics["avg_inventories"].append(summary.get("avg_inventory", 0))
        
        # Compute averages
        results = {
            "avg_total_cost": np.mean(all_metrics["total_costs"]),
            "std_total_cost": np.std(all_metrics["total_costs"]),
            "avg_service_level": np.mean(all_metrics["service_levels"]),
            "avg_stockouts": np.mean(all_metrics["stockouts"]),
            "avg_inventory": np.mean(all_metrics["avg_inventories"]),
            "n_episodes": n_episodes,
        }
        
        eval_env.close()
        return results


def train_rl_agent(
    env_config: Dict[str, Any] = None,
    model_config: Dict[str, Any] = None,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    model_dir: str = "models",
) -> Tuple["PPO", Dict[str, Any]]:
    """
    Convenience function to train RL agent.
    
    Args:
        env_config: Environment configuration
        model_config: Model hyperparameters
        total_timesteps: Training timesteps
        n_envs: Number of parallel environments
        model_dir: Directory to save models
        
    Returns:
        Trained model and info
    """
    trainer = RLTrainer(
        env_params=env_config,
        model_params=model_config,
        model_dir=model_dir,
    )
    
    model, info = trainer.train(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
    )
    
    return model, info
