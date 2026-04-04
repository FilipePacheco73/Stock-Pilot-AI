"""
Interactive Streamlit dashboard for live RL vs manual safety stock control.

This app reuses the existing SupplyChainEnv with minimal changes and runs a
continuous simulation with a rolling 365-day comparison window.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Local imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment.supply_chain_env import SupplyChainEnv


BASE_HOLDING_COST = 0.5
BASE_STOCKOUT_COST = 10.0
BASE_ORDERING_COST = 5.0
WINDOW_DAYS = 365
SCHEDULE_DAYS = 200_000
DEFAULT_SEED = 42
SAFETY_STOCK_MAX = 500
TARGET_DASHBOARD_TRAINING_STEPS = 400_000
TRAIN_SCHEDULE_WINDOW = 30
TRAIN_EPISODE_LENGTH = 365
DASHBOARD_TRAIN_N_ENVS = 8
RL_SAFETY_ADJUSTMENT_MAX = 60.0
RL_COVERAGE_PENALTY = 0.0
RL_SERVICE_BONUS = 0.0
RL_SERVICE_BONUS_THRESHOLD = 0.97


def generate_cost_multiplier_schedule(days: int, seed: int, window: int = TRAIN_SCHEDULE_WINDOW):
    """Generate per-day cost multipliers that change by fixed windows."""
    rng = np.random.RandomState(seed + 9999)
    n_windows = (days + window - 1) // window

    h_mult = rng.uniform(0.2, 3.0, size=n_windows)
    s_mult = rng.uniform(0.2, 3.0, size=n_windows)
    o_mult = rng.uniform(0.2, 3.0, size=n_windows)

    h = np.repeat(h_mult, window)[:days]
    s = np.repeat(s_mult, window)[:days]
    o = np.repeat(o_mult, window)[:days]
    return h, s, o


class ScheduledCostTrainingEnv(SupplyChainEnv):
    """Training env with domain randomization on cost multipliers."""

    def __init__(self, schedule_window: int = TRAIN_SCHEDULE_WINDOW, base_seed: int = DEFAULT_SEED, **kwargs):
        super().__init__(**kwargs)
        self.schedule_window = int(schedule_window)
        self.base_seed = int(base_seed)
        self.episode_idx = 0
        self.h_sched = None
        self.s_sched = None
        self.o_sched = None

    def _apply_day_costs(self):
        if self.h_sched is None:
            return
        day_idx = min(self.current_step, len(self.h_sched) - 1)
        self.holding_cost = float(BASE_HOLDING_COST * self.h_sched[day_idx])
        self.stockout_cost = float(BASE_STOCKOUT_COST * self.s_sched[day_idx])
        self.ordering_cost = float(BASE_ORDERING_COST * self.o_sched[day_idx])

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        episode_seed = self.base_seed + self.episode_idx
        self.h_sched, self.s_sched, self.o_sched = generate_cost_multiplier_schedule(
            days=self.episode_length,
            seed=episode_seed,
            window=self.schedule_window,
        )
        self.episode_idx += 1
        self._apply_day_costs()
        return self._get_state(), info

    def step(self, action):
        # Use current-day costs for transition/reward.
        self._apply_day_costs()
        obs, reward, terminated, truncated, info = super().step(action)
        # Expose next-day costs in next observation for the agent's next decision.
        self._apply_day_costs()
        return self._get_state(), reward, terminated, truncated, info


class StreamlitProgressCallback(BaseCallback):
    """Pushes PPO training progress updates to Streamlit UI."""

    def __init__(self, total_timesteps: int, progress_bar, status_box, message: str, update_every: int = 200):
        super().__init__()
        self.total_timesteps = max(1, int(total_timesteps))
        self.progress_bar = progress_bar
        self.status_box = status_box
        self.message = message
        self.update_every = int(update_every)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_every == 0 or self.num_timesteps >= self.total_timesteps:
            pct = min(1.0, self.num_timesteps / self.total_timesteps)
            self.progress_bar.progress(pct)
            self.status_box.caption(f"{self.message}: {self.num_timesteps:,}/{self.total_timesteps:,} passos")
        return True


class ReplayScenarioEnv(SupplyChainEnv):
    """Environment variant that replays the same exogenous scenario by day."""

    def __init__(self, demand_schedule, lead_time_schedule, **kwargs):
        super().__init__(**kwargs)
        self.demand_schedule = np.asarray(demand_schedule, dtype=np.int32)
        self.lead_time_schedule = np.asarray(lead_time_schedule, dtype=np.int32)

    def _schedule_idx(self) -> int:
        return (self.current_step - 1) % len(self.demand_schedule)

    def _generate_demand(self) -> int:
        return int(self.demand_schedule[self._schedule_idx()])

    def _generate_lead_time(self) -> int:
        return int(self.lead_time_schedule[self._schedule_idx()])


@dataclass
class ScenarioState:
    env_manual: ReplayScenarioEnv
    env_rl: ReplayScenarioEnv
    obs_manual: np.ndarray
    obs_rl: np.ndarray
    day: int
    history: dict


def generate_shared_schedule(days: int, seed: int = DEFAULT_SEED):
    """Generate shared demand and lead-time series used by both scenarios."""
    rng = np.random.RandomState(seed)

    base_demand = rng.normal(50.0, 15.0, size=days)
    spikes = rng.random(size=days) < 0.1
    base_demand[spikes] *= 2.5
    demand = np.maximum(0.0, np.round(base_demand)).astype(np.int32)

    lead_time = rng.randint(1, 6, size=days).astype(np.int32)
    return demand, lead_time


def make_history():
    return {
        "day": deque(maxlen=WINDOW_DAYS),
        "manual_inventory": deque(maxlen=WINDOW_DAYS),
        "manual_safety": deque(maxlen=WINDOW_DAYS),
        "manual_daily_cost": deque(maxlen=WINDOW_DAYS),
        "rl_inventory": deque(maxlen=WINDOW_DAYS),
        "rl_safety": deque(maxlen=WINDOW_DAYS),
        "rl_daily_cost": deque(maxlen=WINDOW_DAYS),
        "holding_cost": deque(maxlen=WINDOW_DAYS),
        "stockout_cost": deque(maxlen=WINDOW_DAYS),
        "ordering_cost": deque(maxlen=WINDOW_DAYS),
    }


def create_env_pair(seed: int = DEFAULT_SEED):
    demand_sched, lead_time_sched = generate_shared_schedule(SCHEDULE_DAYS, seed=seed)

    common_kwargs = {
        "seed": seed,
        "episode_length": 1_000_000,
        "initial_inventory": 100,
        "initial_safety_stock": 300,
        "safety_stock_max": SAFETY_STOCK_MAX,
        "holding_cost": BASE_HOLDING_COST,
        "stockout_cost": BASE_STOCKOUT_COST,
        "ordering_cost": BASE_ORDERING_COST,
        "safety_adjustment_max": RL_SAFETY_ADJUSTMENT_MAX,
        "coverage_penalty_coef": RL_COVERAGE_PENALTY,
        "service_bonus": RL_SERVICE_BONUS,
        "service_bonus_threshold": RL_SERVICE_BONUS_THRESHOLD,
    }

    env_manual = ReplayScenarioEnv(
        demand_schedule=demand_sched,
        lead_time_schedule=lead_time_sched,
        **common_kwargs,
    )
    env_rl = ReplayScenarioEnv(
        demand_schedule=demand_sched,
        lead_time_schedule=lead_time_sched,
        **common_kwargs,
    )

    obs_manual, _ = env_manual.reset(seed=seed)
    obs_rl, _ = env_rl.reset(seed=seed)

    return ScenarioState(
        env_manual=env_manual,
        env_rl=env_rl,
        obs_manual=obs_manual,
        obs_rl=obs_rl,
        day=0,
        history=make_history(),
    )


def load_or_train_model(progress_bar, status_box):
    """Train or refresh the RL policy using the same pure-cost setup as test_visualization."""
    env = make_vec_env(
        lambda: ScheduledCostTrainingEnv(
            seed=DEFAULT_SEED,
            episode_length=TRAIN_EPISODE_LENGTH,
            schedule_window=TRAIN_SCHEDULE_WINDOW,
            base_seed=DEFAULT_SEED,
            safety_stock_max=SAFETY_STOCK_MAX,
            safety_adjustment_max=RL_SAFETY_ADJUSTMENT_MAX,
            coverage_penalty_coef=RL_COVERAGE_PENALTY,
            service_bonus=RL_SERVICE_BONUS,
            service_bonus_threshold=RL_SERVICE_BONUS_THRESHOLD,
        ),
        n_envs=DASHBOARD_TRAIN_N_ENVS,
        seed=DEFAULT_SEED,
    )

    model_candidates = [
        Path("models/live_app_model.zip"),
        Path("models/ppo_agent_training.zip"),
        Path("models/checkpoints/model_period_1.zip"),
        Path("models/test_model.zip"),
    ]

    model = None
    source_label = ""
    for model_path in model_candidates:
        if not model_path.exists():
            continue
        try:
            status_box.caption(f"Carregando modelo: {model_path}")
            progress_bar.progress(0.05)
            model = PPO.load(str(model_path), env=env)
            source_label = f"Checkpoint carregado e retreinado: {model_path.name}"

            train_steps = TARGET_DASHBOARD_TRAINING_STEPS
            progress_cb = StreamlitProgressCallback(
                total_timesteps=train_steps,
                progress_bar=progress_bar,
                status_box=status_box,
                message="Treinando política RL",
                update_every=200,
            )
            model.learn(total_timesteps=train_steps, progress_bar=False, callback=progress_cb)
            progress_bar.progress(1.0)
            break
        except ValueError:
            # Skip incompatible checkpoints (for example, older observation spaces).
            continue

    if model is None:
        status_box.caption("Nenhum checkpoint compatível encontrado. Treinando modelo novo...")
        model = PPO(
            "MlpPolicy",
            env,
            seed=DEFAULT_SEED,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.7,
            verbose=0,
        )
        train_steps = TARGET_DASHBOARD_TRAINING_STEPS
        progress_cb = StreamlitProgressCallback(
            total_timesteps=train_steps,
            progress_bar=progress_bar,
            status_box=status_box,
            message="Treinando política RL",
            update_every=300,
        )
        model.learn(total_timesteps=train_steps, progress_bar=False, callback=progress_cb)
        progress_bar.progress(1.0)
        source_label = "Modelo treinado do zero"

    Path("models").mkdir(parents=True, exist_ok=True)
    model.save("models/live_app_model")
    return model, source_label


def apply_costs(state: ScenarioState, holding_mult: float, stockout_mult: float, ordering_mult: float):
    h = BASE_HOLDING_COST * holding_mult
    s = BASE_STOCKOUT_COST * stockout_mult
    o = BASE_ORDERING_COST * ordering_mult

    for env in (state.env_manual, state.env_rl):
        env.holding_cost = float(h)
        env.stockout_cost = float(s)
        env.ordering_cost = float(o)

    return h, s, o


def run_days(
    state: ScenarioState,
    model: PPO,
    days: int,
    manual_target_safety: int,
    holding_mult: float,
    stockout_mult: float,
    ordering_mult: float,
):
    for _ in range(days):
        h, s, o = apply_costs(state, holding_mult, stockout_mult, ordering_mult)

        # Refresh observations so current multipliers are visible before actions.
        state.obs_manual = state.env_manual._get_state()
        state.obs_rl = state.env_rl._get_state()

        # Manual scenario: user directly controls safety stock target.
        manual_adj = float(manual_target_safety - state.env_manual.safety_stock)
        manual_action = state.env_manual.encode_action(safety_stock_adj=manual_adj)
        state.obs_manual, _, _, _, info_manual = state.env_manual.step(manual_action)

        # RL scenario: trained policy adapts safety stock automatically in real time.
        rl_action, _ = model.predict(state.obs_rl, deterministic=True)
        state.obs_rl, _, _, _, info_rl = state.env_rl.step(rl_action)

        state.day += 1

        state.history["day"].append(state.day)
        state.history["manual_inventory"].append(info_manual["inventory"])
        state.history["manual_safety"].append(info_manual["safety_stock"])
        state.history["manual_daily_cost"].append(
            info_manual["holding_cost"] + info_manual["stockout_cost"] + info_manual["ordering_cost"]
        )
        state.history["rl_inventory"].append(info_rl["inventory"])
        state.history["rl_safety"].append(info_rl["safety_stock"])
        state.history["rl_daily_cost"].append(
            info_rl["holding_cost"] + info_rl["stockout_cost"] + info_rl["ordering_cost"]
        )
        state.history["holding_cost"].append(h)
        state.history["stockout_cost"].append(s)
        state.history["ordering_cost"].append(o)


def render_dashboard(state: ScenarioState):
    if len(state.history["day"]) == 0:
        st.info("Aguardando primeiros passos da simulação...")
        return

    df = pd.DataFrame({k: list(v) for k, v in state.history.items()})

    manual_annual_cost = float(df["manual_daily_cost"].sum())
    rl_annual_cost = float(df["rl_daily_cost"].sum())
    diff = manual_annual_cost - rl_annual_cost
    diff_pct = (diff / manual_annual_cost * 100.0) if manual_annual_cost > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Custo últimos 365 dias (Manual)", f"${manual_annual_cost:,.0f}")
    c2.metric("Custo últimos 365 dias (RL)", f"${rl_annual_cost:,.0f}")
    c3.metric("Diferença RL vs Manual", f"${diff:,.0f}", f"{diff_pct:+.2f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_manual = go.Figure()
        fig_manual.add_trace(
            go.Scatter(x=df["day"], y=df["manual_inventory"], mode="lines", name="Inventário Manual", line=dict(color="#1f77b4")),
        )
        fig_manual.add_trace(
            go.Scatter(x=df["day"], y=df["manual_safety"], mode="lines", name="Safety Stock Manual", line=dict(color="#d62728", dash="dash")),
        )
        fig_manual.update_layout(
            title="Manual: Inventário e Safety Stock",
            xaxis_title="Dia de simulação",
            height=430,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        fig_manual.update_yaxes(title_text="Unidades")
        st.plotly_chart(fig_manual, width="stretch")

    with col_right:
        fig_rl = go.Figure()
        fig_rl.add_trace(
            go.Scatter(x=df["day"], y=df["rl_inventory"], mode="lines", name="Inventário RL", line=dict(color="#17becf")),
        )
        fig_rl.add_trace(
            go.Scatter(x=df["day"], y=df["rl_safety"], mode="lines", name="Safety Stock RL", line=dict(color="#ff7f0e", dash="dash")),
        )
        fig_rl.update_layout(
            title="RL: Inventário e Safety Stock",
            xaxis_title="Dia de simulação",
            height=430,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        fig_rl.update_yaxes(title_text="Unidades")
        st.plotly_chart(fig_rl, width="stretch")

    cost_fig = go.Figure()
    manual_ma = df["manual_daily_cost"].rolling(window=WINDOW_DAYS, min_periods=1).mean()
    rl_ma = df["rl_daily_cost"].rolling(window=WINDOW_DAYS, min_periods=1).mean()
    cost_fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=manual_ma,
            mode="lines",
            name="MM 365d Custo Diário Manual",
            line=dict(color="#1f77b4", width=3),
        )
    )
    cost_fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=rl_ma,
            mode="lines",
            name="MM 365d Custo Diário RL",
            line=dict(color="#2ca02c", width=3),
        )
    )
    cost_fig.update_layout(
        title="Comparação de custo: média móvel de 365 dias (Manual vs RL)",
        xaxis_title="Dia de simulação",
        yaxis_title="Custo médio diário ($)",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    st.plotly_chart(cost_fig, width="stretch")


def main():
    st.set_page_config(page_title="StockPilot Live", layout="wide")

    st.title("StockPilot Live - RL vs Controle Manual")
    st.caption("Simulação contínua com janela móvel de 365 dias e comparação de custo entre cenários.")

    st.warning("Ao iniciar a página, o agente RL é treinado/inicializado antes da simulação ao vivo.")

    if "model" not in st.session_state:
        with st.status("Treinando RL...", expanded=True) as status:
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            model, source_label = load_or_train_model(progress_bar, progress_text)
            st.session_state.model = model
            st.session_state.model_source = source_label
            status.update(label="Treinamento concluído. RL pronto para controle em tempo real.", state="complete")

    model = st.session_state.model
    st.info(f"Status do modelo RL: {st.session_state.get('model_source', 'Modelo ativo')}.")

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = create_env_pair(seed=DEFAULT_SEED)
    if "running" not in st.session_state:
        st.session_state.running = False

    st.sidebar.header("Controles")
    holding_mult = st.sidebar.slider("Multiplicador Holding", 0.2, 3.0, 1.0, 0.05)
    stockout_mult = st.sidebar.slider("Multiplicador Stockout", 0.2, 3.0, 1.0, 0.05)
    ordering_mult = st.sidebar.slider("Multiplicador Ordering", 0.2, 3.0, 1.0, 0.05)
    manual_target_safety = st.sidebar.slider("Safety Stock Manual", 5, SAFETY_STOCK_MAX, 300, 1)
    days_per_tick = st.sidebar.slider("Dias por atualização", 1, 14, 3, 1)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Iniciar"):
        st.session_state.running = True
    if b2.button("Pausar"):
        st.session_state.running = False
    if b3.button("Avançar 1 passo"):
        run_days(
            st.session_state.sim_state,
            model,
            days=1,
            manual_target_safety=manual_target_safety,
            holding_mult=holding_mult,
            stockout_mult=stockout_mult,
            ordering_mult=ordering_mult,
        )
    if b4.button("Resetar"):
        st.session_state.sim_state = create_env_pair(seed=DEFAULT_SEED)
        st.session_state.running = False

    if st.session_state.running:
        run_days(
            st.session_state.sim_state,
            model,
            days=days_per_tick,
            manual_target_safety=manual_target_safety,
            holding_mult=holding_mult,
            stockout_mult=stockout_mult,
            ordering_mult=ordering_mult,
        )

    st.markdown(
        f"**Dia atual:** {st.session_state.sim_state.day} | "
        f"**Modo:** {'Rodando' if st.session_state.running else 'Pausado'} | "
        f"**Janela:** últimos {WINDOW_DAYS} dias"
    )

    render_dashboard(st.session_state.sim_state)

    if st.session_state.running:
        time.sleep(0.8)
        st.rerun()


if __name__ == "__main__":
    main()
