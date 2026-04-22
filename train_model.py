import argparse
import matplotlib
import yaml
import pandas as pd
import logging
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.trading_env import TradingEnv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from src.buy_and_hold import buy_and_hold
from evaluation.evaluation_metrics import calculate_sharpe, calculate_max_drawdown,volatility, num_trades, total_returns, win_rate, calmar_ratio, calculate_final_net_worth, annualized_return
from evaluation.agent_metrics import (
    prob_up,
    prob_max_drawdown,
    signal_entropy,
    macd,
    relative_strength,
    ddi,
    rolling_volatility

)



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration.

    Raises:
        yaml.YAMLError: If the YAML file is invalid.
        Exception: For other file loading errors.
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            logger.info(f"YAML content:\n{content}")
            return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RL trading model with/without sentiment")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger.info(f"Loaded config: {config}")

symbol = config["stock_symbol"]
initial_balance = config.get("initial_balance", 10000)
start_date = config["start_date"]
end_date = config["end_date"]


raw_dir = config["raw_dir"]
processed_dir = config["processed_dir"]
sentiment_mode = config.get("sentiment_mode", "individual")
sentiment_source = config.get("sentiment_source", "finnhub_orig")
algo_name = config.get("algo", "PPO")
use_lstm = config.get("use_lstm", False)
lstm_window = config.get("lstm_window", 32)
lstm_hidden_size = config.get("lstm_hidden_size", 64)
train_test_split = config.get("train_test_split", 0.7)
replicates = config.get("replicates", 1)


# === DATA PATH (CORREGIDO) ===
# === DATA PATH (CORREGIDO) ===
raw_csv = os.path.join(raw_dir, f"{symbol}_raw.csv")
processed_csv = os.path.join(processed_dir, f"{symbol}_sentiment_{sentiment_source if sentiment_mode == 'individual' else 'combined'}.csv")
ensemble_csv = os.path.join(processed_dir, f"{symbol}_hybrid_ready.csv")

# Selección con prioridad: 1. Ensemble, 2. Sentiment, 3. Raw
if os.path.exists(ensemble_csv):
    data_path = ensemble_csv
    logger.info(f"Using Hybrid Ensemble data: {data_path}")
elif os.path.exists(processed_csv):
    data_path = processed_csv
    logger.info(f"Using Sentiment data: {data_path}")
else:
    data_path = raw_csv
    logger.warning("No Hybrid or Sentiment CSV found. Using RAW data.")

# Cargamos el dataframe una sola vez
df = pd.read_csv(data_path)
logger.info(f"Successfully loaded {len(df)} rows from: {data_path}")


# Detectar columna de fecha automáticamente
if 'date' in df.columns:
    date_col = 'date'
elif 'Date' in df.columns:
    date_col = 'Date'
else:
    raise ValueError(f"No date column found. Columns: {df.columns}")

#ahora el índice del dataframe es el tiempo
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

# Rellenar días faltantes
df = df.asfreq('B')
df.ffill(inplace=True)
df.fillna(0, inplace=True)
start_dt = pd.to_datetime(start_date)

if end_date is None:
    df = df[df.index >= start_dt].copy()
else:
    end_dt = pd.to_datetime(end_date)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()

df = df.sort_index()
df.name = symbol
logger.info(f"Filtered data: {len(df)} rows from {df.index.min()} to {df.index.max()}")



# Fill any NaNs with 0 (important for the first row or edge cases)
df["prob_up"] = prob_up(df["close"], horizon=1)
df["prob_max_drawdown"] = prob_max_drawdown(df["close"], horizon=1, threshold=0.1)
df["signal_entropy"] = signal_entropy(df["close"], horizon=1)
df["macd"] = macd(df["close"])          # MACD diario
df["rsi"] = relative_strength(df["close"])
df["ddi"] = ddi(df["high"], df["low"], df["close"])
df["rolling_vol"] = rolling_volatility(df["close"])
#rellenamos con 0 al principio
df[["prob_up","prob_max_drawdown","signal_entropy"]] = (df[["prob_up","prob_max_drawdown","signal_entropy"]].fillna(0.0))
df[["macd","rsi","ddi","rolling_vol"]] = df[["macd","rsi","ddi","rolling_vol"]].fillna(0.0) 

# --- DIVISIÓN DE DATOS (TRAIN / TEST) ---
# Supongamos un 80% para entrenar y un 20% para testear
split_percentage = 0.8
split_index = int(len(df) * split_percentage)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

logger.info(f"Entrenamiento: {len(train_df)} días ({train_df.index.min()} a {train_df.index.max()})")
logger.info(f"Test/Evaluación: {len(test_df)} días ({test_df.index.min()} a {test_df.index.max()})")


# ================================
# ENVIRONMENTS (CLAROS Y SEPARADOS)
# ================================

# 1. RL SIN sentimiento (baseline puro)
vec_env_rl_no_sent = make_vec_env(
    lambda: TradingEnv(train_df, use_sentiment=False, use_ensemble=False),
    n_envs=1
)

# 2. RL CON sentimiento
vec_env_rl_sent = make_vec_env(
    lambda: TradingEnv(train_df, use_sentiment=True, use_ensemble=False),
    n_envs=1
)

# 3. RL + Ensemble SIN sentimiento
vec_env_rl_ens_no_sent = make_vec_env(
    lambda: TradingEnv(train_df, use_sentiment=False, use_ensemble=True),
    n_envs=1
)

# 4. RL + Ensemble CON sentimiento (tu modelo "final")
vec_env_rl_ens_sent = make_vec_env(
    lambda: TradingEnv(train_df, use_sentiment=True, use_ensemble=True),
    n_envs=1
)


# ================================
# DEVICE
# ================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")


# ================================
# MODELOS PPO
# ================================

# Paths claros (clave para no liarte)
model_path_rl_no_sent = "models/ppo_rl_no_sent"
model_path_rl_sent = "models/ppo_rl_sent"
model_path_rl_ens_no_sent = "models/ppo_rl_ens_no_sent"
model_path_rl_ens_sent = "models/ppo_rl_ens_sent"


def train_or_load(model_path, env, name):
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        model.set_env(env)
        logger.info(f"Continuing training {name}")
        model.learn(total_timesteps=100_000)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=0.0001,
            clip_range=0.2,
            ent_coef=0.01
        )
        logger.info(f"Training {name} for {config['timesteps']} timesteps")
        model.learn(total_timesteps=config['timesteps'])

    model.save(model_path)
    return model


# Entrenamiento (MISMA lógica, solo ordenado)
model_rl_no_sent = train_or_load(model_path_rl_no_sent, vec_env_rl_no_sent, "RL no sentiment")
model_rl_sent = train_or_load(model_path_rl_sent, vec_env_rl_sent, "RL with sentiment")
model_rl_ens_no_sent = train_or_load(model_path_rl_ens_no_sent, vec_env_rl_ens_no_sent, "RL + Ensemble no sentiment")
model_rl_ens_sent = train_or_load(model_path_rl_ens_sent, vec_env_rl_ens_sent, "RL + Ensemble with sentiment")


# ================================
# DATAFRAMES DE SIMULACIÓN
# ================================

simulation_rl_no_sent = pd.DataFrame(index=test_df.index, columns=['net_worth', 'action'])
simulation_rl_sent = pd.DataFrame(index=test_df.index, columns=['net_worth', 'action'])
simulation_rl_ens_no_sent = pd.DataFrame(index=test_df.index, columns=['net_worth', 'action'])
simulation_rl_ens_sent = pd.DataFrame(index=test_df.index, columns=['net_worth', 'action'])


# ================================
# FUNCIÓN DE SIMULACIÓN (NO CAMBIA LÓGICA)
# ================================

def run_simulation(model, env_config, sim_df):
    env = TradingEnv(test_df, **env_config)
    obs = env.reset()[0]

    for step in range(len(test_df) - env.window_size):
        action, _ = model.predict(obs)
        action_val = action.item()

        obs, reward, done, truncated, info = env.step(action_val)

        current_idx = env.current_step - 1
        date = test_df.index[current_idx]
        net_worth_value = env.net_worth

        sim_df.loc[date, 'net_worth'] = net_worth_value
        sim_df.loc[date, 'action'] = action_val

        if done or truncated:
            break

    # fill
    sim_df['net_worth'] = sim_df['net_worth'].fillna(initial_balance).ffill()
    sim_df['action'] = sim_df['action'].fillna(0)

    return sim_df


# ================================
# EJECUCIÓN DE SIMULACIONES
# ================================

simulation_rl_no_sent = run_simulation(
    model_rl_no_sent,
    {"use_sentiment": False, "use_ensemble": False},
    simulation_rl_no_sent
)

simulation_rl_sent = run_simulation(
    model_rl_sent,
    {"use_sentiment": True, "use_ensemble": False},
    simulation_rl_sent
)

simulation_rl_ens_no_sent = run_simulation(
    model_rl_ens_no_sent,
    {"use_sentiment": False, "use_ensemble": True},
    simulation_rl_ens_no_sent
)

simulation_rl_ens_sent = run_simulation(
    model_rl_ens_sent,
    {"use_sentiment": True, "use_ensemble": True},
    simulation_rl_ens_sent
)


# ================================
# BUY & HOLD (igual)
# ================================
simulation_df_bh, metrics_bh, actions_bh, final_buy_hold = buy_and_hold(test_df, initial_balance=initial_balance)

# ============================================
# LIMPIEZA Y VALIDACIÓN (COHERENTE CON NUEVOS NOMBRES)
# ============================================

all_simulations = [
    simulation_rl_no_sent,
    simulation_rl_sent,
    simulation_rl_ens_no_sent,
    simulation_rl_ens_sent
]

for sim_df in all_simulations:
    # 1. Rellenar ventana inicial
    sim_df['net_worth'] = sim_df['net_worth'].fillna(initial_balance)
    sim_df['action'] = sim_df['action'].fillna(0)

    # 2. Forward fill por si corta antes
    sim_df['net_worth'] = sim_df['net_worth'].ffill()
    sim_df['action'] = sim_df['action'].fillna(0)

    # 3. Tipos correctos (evita warnings)
    sim_df['net_worth'] = pd.to_numeric(sim_df['net_worth'])
    sim_df['action'] = pd.to_numeric(sim_df['action']).astype(int)

logger.info("Date indices and data aligned successfully")


# ============================================
# VALIDACIONES
# ============================================

for name, sim_df in {
    "rl_no_sent": simulation_rl_no_sent,
    "rl_sent": simulation_rl_sent,
    "rl_ens_no_sent": simulation_rl_ens_no_sent,
    "rl_ens_sent": simulation_rl_ens_sent
}.items():
    
    if not sim_df.index.equals(test_df.index):
        logger.error(f"Index misalignment in {name}")
        raise ValueError(f"Index misalignment in {name}")
        
    if sim_df['net_worth'].isna().any():
        logger.error(f"Missing net_worth values in {name}")
        raise ValueError(f"Missing net_worth values in {name}")

logger.info("All simulations validated correctly")


# ============================================
# MÉTRICAS PRINCIPALES (modelo final vs baseline)
# ============================================

# Modelo final (ensemble + sentiment)



# ============================================
# PREDICCIÓN PARA MAÑANA
# ============================================

env = TradingEnv(test_df, use_sentiment=True, use_ensemble=True)
obs = env.reset()[0]

done = False
while not done:
    obs, _, done, _, _ = env.step(0)

action_tomorrow, _ = model_rl_ens_sent.predict(obs)
action_tomorrow = action_tomorrow.item()

action_map = {0: "HOLD ⚪", 1: "BUY 🟢", 2: "SELL 🔴"}
print(f"\nRecomendación para mañana (Modelo final): {action_map[action_tomorrow]}")


# ============================================
# LOGS
# ============================================



# ============================================
# SAVE RESULTS
# ============================================

results_df = pd.DataFrame({
    'Date': test_df.index,
    'Net_Worth_Final': simulation_rl_ens_sent['net_worth'],
    'Net_Worth_Baseline': simulation_rl_sent['net_worth'],
    'Net_Worth_No_Sent': simulation_rl_no_sent['net_worth'],
    'Net_Worth_Ens_No_Sent': simulation_rl_ens_no_sent['net_worth'],
    'Actions_Final': simulation_rl_ens_sent['action'],
})

os.makedirs('results', exist_ok=True)
results_df['Net_Worth_Buy_Hold'] = simulation_df_bh['net_worth']

results_df.to_csv('results/aapl_trading_results.csv', index=False)
logger.info("Saved trading results to results/aapl_trading_results.csv")


# ============================================
# RETURNS PARA MÉTRICAS
# ============================================

returns_final = simulation_rl_ens_sent['net_worth'].pct_change().dropna()
returns_baseline = simulation_rl_sent['net_worth'].pct_change().dropna()
returns_bh = simulation_df_bh['net_worth'].pct_change().dropna()


# ============================================
# SUMMARY
# ============================================

# ============================================
# SUMMARY (TODOS LOS MODELOS)
# ============================================

simulations_dict = {
    "RL": simulation_rl_no_sent,
    "RL+Sent": simulation_rl_sent,
    "RL+Ens": simulation_rl_ens_no_sent,
    "RL+Ens+Sent": simulation_rl_ens_sent,
    "Buy_Hold": simulation_df_bh
}

actions_dict = {
    "RL": simulation_rl_no_sent['action'],
    "RL+Sent": simulation_rl_sent['action'],
    "RL+Ens": simulation_rl_ens_no_sent['action'],
    "RL+Ens+Sent": simulation_rl_ens_sent['action'],
    "Buy_Hold": actions_bh
}

metrics_summary = {
    "Metric": [
        "Final Net Worth", "Sharpe Ratio", "Volatility", "Max Drawdown",
        "Total Return", "Win Rate", "Calmar Ratio", "Number of trades", "Annualized returns"
    ]
}

# Construcción dinámica
for name, sim in simulations_dict.items():
    
    returns = sim['net_worth'].pct_change().dropna()

    metrics_summary[name] = [
        sim['net_worth'].iloc[-1],
        calculate_sharpe(sim['net_worth'], freq="1d"),
        volatility(returns),
        calculate_max_drawdown(sim['net_worth']),
        total_returns(sim['net_worth']),
        win_rate(sim['net_worth'], actions_dict[name]),
        calmar_ratio(sim['net_worth']),
        num_trades(actions_dict[name]),
        annualized_return(sim['net_worth'])
    ]

# DataFrame final
metrics_df = pd.DataFrame(metrics_summary)

print("\n=== Trading Metrics Summary ===")
print(metrics_df.to_string(index=False))

# Guardado
os.makedirs("results", exist_ok=True)
metrics_df.to_csv("results/trading_metrics_summary.csv", index=False)

logger.info("Saved metrics summary")

def plot_main_results(test_df, simulations_dict, simulation_df_bh, initial_balance, symbol):
    """
    simulations_dict = {
        "RL": simulation_rl_no_sent,
        "RL+Sent": simulation_rl_sent,
        "RL+Ens": simulation_rl_ens_no_sent,
        "RL+Ens+Sent": simulation_rl_ens_sent
    }
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    # =========================
    # 1. PRECIO + ACCIONES (solo modelo final)
    # =========================
    final_model = simulations_dict["RL+Ens+Sent"]

    common_index = final_model.index.intersection(test_df.index)
    test_df_plot = test_df.loc[common_index]
    final_model = final_model.loc[common_index]

    ax1.plot(test_df_plot.index, test_df_plot['close'], label='Close Price', alpha=0.7)

    buy_mask = final_model['action'].values == 1
    sell_mask = final_model['action'].values == 2

    buy_points = test_df_plot[buy_mask]
    sell_points = test_df_plot[sell_mask]

    ax1.scatter(buy_points.index, buy_points['close'], marker='^', label='Buy', s=60)
    ax1.scatter(sell_points.index, sell_points['close'], marker='v', label='Sell', s=60)

    ax1.set_title(f"{symbol} Price & Actions (Final Model)")
    ax1.legend()

    # =========================
    # 2. NET WORTH (todos)
    # =========================
    for name, sim in simulations_dict.items():
        ax2.plot(sim.index, sim['net_worth'], label=name)

    ax2.plot(simulation_df_bh.index, simulation_df_bh['net_worth'],
             label="Buy & Hold", linestyle=':')

    ax2.set_title("Net Worth Comparison")
    ax2.set_ylabel("Net Worth ($)")
    ax2.legend()

    # =========================
    # 3. P&L (todos)
    # =========================
    for name, sim in simulations_dict.items():
        pnl = sim['net_worth'] - initial_balance
        ax3.plot(sim.index, pnl, label=name)

    pnl_bh = simulation_df_bh['net_worth'] - initial_balance
    ax3.plot(simulation_df_bh.index, pnl_bh, label="Buy & Hold", linestyle=':')

    ax3.axhline(0, linestyle='--')

    ax3.set_title("Profit & Loss")
    ax3.set_ylabel("P&L ($)")
    ax3.set_xlabel("Date")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f"results/{symbol}_main_plots.png", dpi=150)
    plt.show()


def plot_returns_distribution(simulations_dict):
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (name, sim) in zip(axes, simulations_dict.items()):
        returns = sim['net_worth'].pct_change()

        df_analysis = pd.DataFrame({
            "returns": returns,
            "action": sim['action']
        }).dropna()

        sns.boxplot(x="action", y="returns", data=df_analysis, ax=ax)

        ax.set_title(name)
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['HOLD', 'BUY', 'SELL'])

    plt.suptitle("Returns Distribution by Action")
    plt.tight_layout()
    plt.show()


def plot_action_distributions(simulations_dict):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes = axes.flatten()

    for ax, (name, sim) in zip(axes, simulations_dict.items()):
        sns.countplot(x=sim['action'].astype(int), ax=ax)
        ax.set_title(name)
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['HOLD', 'BUY', 'SELL'])

    plt.suptitle("Action Distribution per Model")
    plt.tight_layout()
    plt.show()
simulations_dict = {
    "RL": simulation_rl_no_sent,
    "RL+Sent": simulation_rl_sent,
    "RL+Ens": simulation_rl_ens_no_sent,
    "RL+Ens+Sent": simulation_rl_ens_sent
}

plot_main_results(test_df, simulations_dict, simulation_df_bh, initial_balance, symbol)

plot_action_distributions(simulations_dict)

plot_returns_distribution(simulations_dict)