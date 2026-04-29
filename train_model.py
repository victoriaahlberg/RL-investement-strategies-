import argparse
import matplotlib
import yaml
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import logging
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
#from src.trading_env import TradingEnv
from src.trading_env_global import TradingEnvGlobal
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
    lambda: TradingEnvGlobal(train_df, use_sentiment=False, use_ensemble=False),
    n_envs=1
)

# 2. RL CON sentimiento
vec_env_rl_sent = make_vec_env(
    lambda: TradingEnvGlobal(train_df, use_sentiment=True, use_ensemble=False),
    n_envs=1
)

# 3. RL + Ensemble SIN sentimiento
vec_env_rl_ens_no_sent = make_vec_env(
    lambda: TradingEnvGlobal(train_df, use_sentiment=False, use_ensemble=True),
    n_envs=1
)

# 4. RL + Ensemble CON sentimiento (tu modelo "final")
vec_env_rl_ens_sent = make_vec_env(
    lambda: TradingEnvGlobal(train_df, use_sentiment=True, use_ensemble=True),
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

# Paths claros

model_path_rl_no_sent = f"models/{symbol}_ppo_rl_no_sent"
model_path_rl_sent = f"models/{symbol}_ppo_rl_sent"
model_path_rl_ens_no_sent = f"models/{symbol}_ppo_rl_ens_no_sent"
model_path_rl_ens_sent = f"models/{symbol}_ppo_rl_ens_sent"

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
    env = TradingEnvGlobal(test_df, **env_config)
    obs, _ = env.reset()
    for step in range(len(test_df) - env.window_size):
        action, _ = model.predict(obs, deterministic=False) #ahora cada ejecucion genrea trayectorias difrentes 
        action_val = action.item()

        obs, reward, done, truncated, info = env.step(action_val)

        net_worth_value = info["net_worth"]  # ✔ correcto

        #current_idx = env.step_idx - 1
        date = env.original_index[env.step_idx - 1]

        sim_df.loc[date, 'net_worth'] = net_worth_value
        sim_df.loc[date, 'action'] = action_val
        #date = test_df.index[step]
        

        if done or truncated:
            break

    # fill
    sim_df['net_worth'] = sim_df['net_worth'].fillna(initial_balance)
    sim_df['action'] = pd.to_numeric(sim_df['action'], errors='coerce').fillna(0).astype(int)

    return sim_df

def run_montecarlo(model, env_config, n_runs=100):

    results = []

    for i in range(n_runs):

        env = TradingEnvGlobal(test_df, **env_config)
        obs, _ = env.reset()
        sim_df = pd.DataFrame(index=test_df.index, columns=["net_worth", "action"])

        done = False
        t = 0

        # estado inicial
        sim_df.loc[t, "net_worth"] = initial_balance
        sim_df.loc[t, "action"] = 0

        while not done:

            action, _ = model.predict(obs, deterministic=False)
            action_val = int(action.item())

            obs, reward, done, truncated, info = env.step(action_val)
            if t >= len(test_df):
                break

            date = test_df.index[t]
            sim_df.loc[date, "net_worth"] = float(info["net_worth"])
            sim_df.loc[date, "action"] = action_val
            t+=1
           

            if truncated:
                break

        # limpieza
        sim_df = sim_df.ffill().fillna(initial_balance)

        sim_df["action"] = pd.to_numeric(sim_df["action"], errors="coerce") \
            .fillna(0).astype(int)

        results.append(sim_df)
        

    return results

# ================================
# EJECUCIÓN DE SIMULACIONES
# ================================
# ================================
# MONTE CARLO SIMULATIONS
# ================================


N_MC = 100
 
mc_rl_no_sent = run_montecarlo(
    model_rl_no_sent,
    {"use_sentiment": False, "use_ensemble": False},
    N_MC
)
 
mc_rl_sent = run_montecarlo(
    model_rl_sent,
    {"use_sentiment": True, "use_ensemble": False},
    N_MC
)
 
mc_rl_ens_no_sent = run_montecarlo(
    model_rl_ens_no_sent,
    {"use_sentiment": False, "use_ensemble": True},
    N_MC
)
 
mc_rl_ens_sent = run_montecarlo(
    model_rl_ens_sent,
    {"use_sentiment": True, "use_ensemble": True},
    N_MC
)
 
logger.info(f"MC runs complete: {N_MC} trajectories per model")
 
# ================================
# BUG 1 FIX — derive simulation DataFrames from MC mean trajectories
# (previously these were never populated, leaving all net_worth = 10000)
# ================================

def mc_to_mean_df(mc_runs, reference_index):
    net_worth_cols = []
    action_cols    = []

    for r in mc_runs:
        sim = r.copy().reindex(reference_index).ffill()
        net_worth_cols.append(sim["net_worth"].to_numpy())
        action_cols.append(
            pd.to_numeric(sim["action"], errors="coerce").fillna(0).astype(int).to_numpy()
        )

    nw_arr  = np.vstack(net_worth_cols)       # shape (n_runs, n_steps)
    act_arr = np.vstack(action_cols)           # shape (n_runs, n_steps)

    mean_nw = nw_arr.mean(axis=0)

    # majority vote per timestep: most common action across all runs
    majority_action = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=3).argmax(),
        axis=0,
        arr=act_arr
    )

    return pd.DataFrame(
        {"net_worth": mean_nw, "action": majority_action},
        index=reference_index
    )

simulation_rl_no_sent     = mc_to_mean_df(mc_rl_no_sent,     test_df.index)
simulation_rl_sent        = mc_to_mean_df(mc_rl_sent,        test_df.index)
simulation_rl_ens_no_sent = mc_to_mean_df(mc_rl_ens_no_sent, test_df.index)
simulation_rl_ens_sent    = mc_to_mean_df(mc_rl_ens_sent,    test_df.index)
 
simulations_dict = {
    "RL":          simulation_rl_no_sent,
    "RL+Sent":     simulation_rl_sent,
    "RL+Ens":      simulation_rl_ens_no_sent,
    "RL+Ens+Sent": simulation_rl_ens_sent,
}
 
mc_dict = {
    "RL":          mc_rl_no_sent,
    "RL+Sent":     mc_rl_sent,
    "RL+Ens":      mc_rl_ens_no_sent,
    "RL+Ens+Sent": mc_rl_ens_sent,
}
 
logger.info("Simulation DataFrames built from MC mean trajectories")
 
# ================================
# VALIDATION — log a sanity check per model
# ================================
 
for name, sim in simulations_dict.items():
    final_nw = sim["net_worth"].iloc[-1]
    n_trades  = sim["action"].value_counts().to_dict()
    logger.info(f"{name} | final net worth: {final_nw:.2f} | action counts: {n_trades}")
 
# ================================
# BUY & HOLD BASELINE
# ================================
 
simulation_df_bh, metrics_bh, actions_bh, final_buy_hold = buy_and_hold(
    test_df, initial_balance=initial_balance
)
 
# ================================
# MC METRICS (with BUG 2 fix — calmar_ratio now actually appended)
# ================================
 
def mc_metrics_summary(mc_runs, reference_index=None):
    """
    Compute metrics two ways:
    - Distribution stats (CI, skew, kurtosis) from all N MC runs
    - Point metrics (sharpe, vol, etc.) from the MEAN trajectory
      so they reflect the expected behaviour, not noise from short runs
    """
    if reference_index is None:
        reference_index = test_df.index

    # --- distribution of final wealth across runs ---
    final_values = np.array([
    pd.to_numeric(
        r.reindex(reference_index)["net_worth"],
        errors="coerce"
    ).ffill().iloc[-1]
    for r in mc_runs
    ])
    
    sharpe_values = np.array([
    calculate_sharpe(
        pd.to_numeric(
            r.reindex(reference_index)["net_worth"],
            errors="coerce"
        ).ffill(),
        freq="1d"
    )
    for r in mc_runs
    ])


    # --- mean trajectory (same logic as mc_to_mean_df) ---
    mean_sim = mc_to_mean_df(mc_runs, reference_index)
    w        = mean_sim["net_worth"].astype(float)
    actions  = mean_sim["action"]
    returns  = w.pct_change(fill_method=None).dropna()

    return {
        # Performance
  
        # Returns  (computed on mean trajectory — no more zeros)
        "total_return_mean":  total_returns(w),
        "total_return_std":   np.std([total_returns(
                                  mc_to_mean_df([r], reference_index)["net_worth"]
                              ) for r in mc_runs]),
    # --- Net Worth ---
        "final_net_worth_mean": np.mean(final_values),
        "final_net_worth_std": np.std(final_values),
        "final_net_worth_ci_low": np.percentile(final_values, 2.5),
        "final_net_worth_ci_high": np.percentile(final_values, 97.5),
        "skew_final_wealth": pd.Series(final_values).skew(),
        "kurtosis_final_wealth": pd.Series(final_values).kurtosis(),

        # --- Sharpe ---
        "sharpe_mean": np.mean(sharpe_values),
        "sharpe_std": np.std(sharpe_values),
        "sharpe_ci_low": np.percentile(sharpe_values, 2.5),
        "sharpe_ci_high": np.percentile(sharpe_values, 97.5),
        "skew_sharpe": pd.Series(sharpe_values).skew(),
        "kurtosis_sharpe": pd.Series(sharpe_values).kurtosis(),

    # --- resto igual ---
        "annual_return_mean": annualized_return(w),
        "annual_return_std":  0.0,   # expensive to recompute; set 0 or remove 
        "volatility_mean":    volatility(returns),
        "volatility_std":     0.0,
        "max_drawdown_mean":  calculate_max_drawdown(w),
        "max_drawdown_worst": np.min([calculate_max_drawdown(
                                  r["net_worth"].astype(float).ffill()
                              ) for r in mc_runs]),
        "calmar_mean":        calmar_ratio(w),
        "calmar_std":         0.0,
        # Behaviour
        "num_trades_mean": num_trades(actions),
        "num_trades_std":  np.std([num_trades(
                               pd.to_numeric(r["action"], errors="coerce").fillna(0).astype(int)
                           ) for r in mc_runs]),
        "win_rate_mean":   win_rate(w, actions),
        "win_rate_std":    0.0,
        # Distribution shape
        "skew_final_wealth":     pd.Series(final_values).skew(),
        "kurtosis_final_wealth": pd.Series(final_values).kurtosis(),
    }


 
 
results = {name: mc_metrics_summary(runs) for name, runs in mc_dict.items()}

stats_table = pd.DataFrame({
    "Metric": [
        "Net Worth Mean",
        "Net Worth Std",
        "Net Worth CI Low",
        "Net Worth CI High",
        "Net Worth Skew",
        "Net Worth Kurtosis",
        "Sharpe Mean (MC)",
        "Sharpe Std",
        "Sharpe CI Low",
        "Sharpe CI High",
        "Sharpe Skew",
        "Sharpe Kurtosis",
    ],
    **{
        name: [
            results[name]["final_net_worth_mean"],
            results[name]["final_net_worth_std"],
            results[name]["final_net_worth_ci_low"],
            results[name]["final_net_worth_ci_high"],
            results[name]["skew_final_wealth"],
            results[name]["kurtosis_final_wealth"],
            results[name]["sharpe_mean"],
            results[name]["sharpe_std"],
            results[name]["sharpe_ci_low"],
            results[name]["sharpe_ci_high"],
            results[name]["skew_sharpe"],
            results[name]["kurtosis_sharpe"],
        ]
        for name in ["RL", "RL+Sent", "RL+Ens", "RL+Ens+Sent"]
    }
})
styled = stats_table.style \
    .format("{:.4f}") \
    .background_gradient(cmap="RdYlGn", axis=1)

print(styled)
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

table = ax.table(
    cellText=stats_table.round(3).values,
    colLabels=stats_table.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("Monte Carlo Statistical Summary")
plt.savefig(f"results/{symbol}_stats_table.png", dpi=150)
plt.show()
 
# ================================
# METRICS TABLE
# ================================
 
bh_final   = simulation_df_bh["net_worth"].iloc[-1]
bh_returns = simulation_df_bh["net_worth"].pct_change(fill_method=None).dropna()  # DEPRECATION FIX
 
metrics_table = pd.DataFrame({
    "Metric": [
        "Final Net Worth", "Sharpe", "Volatility", "Max Drawdown",
        "Total Return", "Annual Return", "Calmar", "Num Trades", "Win Rate",
    ],
    **{
        name: [
            results[name]["final_net_worth_mean"],
            results[name]["sharpe_mean"],
            results[name]["volatility_mean"],
            results[name]["max_drawdown_mean"],
            results[name]["total_return_mean"],
            results[name]["annual_return_mean"],
            results[name]["calmar_mean"],       # now populated
            results[name]["num_trades_mean"],
            results[name]["win_rate_mean"],
        ]
        for name in ["RL", "RL+Sent", "RL+Ens", "RL+Ens+Sent"]
    },
    "Buy and Hold": [
        bh_final,
        calculate_sharpe(simulation_df_bh["net_worth"], freq="1d"),
        volatility(bh_returns),
        calculate_max_drawdown(simulation_df_bh["net_worth"]),
        (bh_final / simulation_df_bh["net_worth"].iloc[0]) - 1,
        annualized_return(simulation_df_bh["net_worth"]),
        np.nan,   # calmar not meaningful for single-trade B&H
        1,
        np.nan,
    ],
})
 
print(metrics_table.to_string(index=False))
 
# ================================
# TOMORROW'S PREDICTION
# ================================
 
env = TradingEnvGlobal(test_df, use_sentiment=True, use_ensemble=True)
obs = env.reset()[0]
 
done = False
while not done:
    obs, _, done, _, _ = env.step(0)
 
action_tomorrow, _ = model_rl_ens_sent.predict(obs)
action_tomorrow    = action_tomorrow.item()
 
action_map = {0: "HOLD ⚪", 1: "BUY 🟢", 2: "SELL 🔴"}
print(f"\nRecomendación para mañana (Modelo final): {action_map[action_tomorrow]}")
 
# ================================
# SAVE RESULTS (with DEPRECATION FIX on infer_objects)
# ================================
 
os.makedirs("results", exist_ok=True)
aligned_index = test_df.index
 
results_df = pd.DataFrame(index=aligned_index)
 
results_df["Net_Worth_Final"]       = (simulation_rl_ens_sent["net_worth"]
                                        .reindex(aligned_index).ffill()
                                        .infer_objects(copy=False))  # DEPRECATION FIX
results_df["Net_Worth_RL_Sent"]     = (simulation_rl_sent["net_worth"]
                                        .reindex(aligned_index).ffill()
                                        .infer_objects(copy=False))
results_df["Net_Worth_RL_No_Sent"]  = (simulation_rl_no_sent["net_worth"]
                                        .reindex(aligned_index).ffill()
                                        .infer_objects(copy=False))
results_df["Net_Worth_RL_Ens_No_Sent"] = (simulation_rl_ens_no_sent["net_worth"]
                                            .reindex(aligned_index).ffill()
                                            .infer_objects(copy=False))
results_df["Actions_RL_Ens_Sent"]   = (simulation_rl_ens_sent["action"]
                                        .reindex(aligned_index)
                                        .fillna(0).infer_objects(copy=False).astype(int))
results_df["Net_Worth_Buy_Hold"]    = (simulation_df_bh["net_worth"]
                                        .reindex(aligned_index).ffill()
                                        .infer_objects(copy=False))
 
results_df.to_csv("results/aapl_trading_results.csv")
logger.info("Saved trading results to results/aapl_trading_results.csv")


def align(sim):
    return sim.reindex(test_df.index).ffill()

simulations_dict = {
        "RL": simulation_rl_no_sent,
        "RL+Sent": simulation_rl_sent,
        "RL+Ens": simulation_rl_ens_no_sent,
        "RL+Ens+Sent": simulation_rl_ens_sent
    }

def plot_main_results(test_df, simulations_dict, simulation_df_bh, initial_balance, symbol):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    # =========================
    # 1. PRECIO + ACCIONES (solo modelo final)
    # =========================
    final_model = simulation_rl_ens_sent.reindex(test_df.index).ffill()

    common_index = final_model.index.intersection(test_df.index)
    test_df_plot = test_df.loc[common_index]
    final_model = final_model.loc[common_index]


    ax1.plot(test_df.index, test_df["close"], label="Close Price")

    buy = final_model["action"].reindex(test_df.index).fillna(0) == 1
    sell = final_model["action"].reindex(test_df.index).fillna(0) == 2

    ax1.scatter(test_df.index[buy], test_df["close"][buy], marker="^")
    ax1.scatter(test_df.index[sell], test_df["close"][sell], marker="v")

    ax1.set_title(f"{symbol} Price & Actions (Final Model)")
    ax1.legend()

    # =========================
    # 2. NET WORTH (todos)
    # =========================
    for name, sim in simulations_dict.items():
        sim = sim.reindex(test_df.index).ffill()
        ax2.plot(sim.index, sim["net_worth"], label=name)

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



def plot_secondary_dashboard(simulations_dict, mc_dict, initial_balance, symbol):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- 1. Action distribution (from MC runs, not mean sim) ---
    ax = axes[0, 0]
    for name, runs in mc_dict.items():
        all_actions = np.concatenate([
            pd.to_numeric(r["action"], errors="coerce").fillna(0).astype(int).to_numpy()
            for r in runs
        ])
        if len(np.unique(all_actions)) > 1:
            sns.kdeplot(all_actions, label=name, ax=ax, warn_singular=False)
        else:
            ax.axvline(all_actions[0], label=f"{name} (constant)", linestyle="--")
    ax.set_title("Action distribution (pooled MC runs)")
    ax.set_xlabel("Action  (0=Hold, 1=Buy, 2=Sell)")
    ax.legend()

    # --- 2. Returns distribution ---
    ax = axes[0, 1]
    for name, sim in simulations_dict.items():
        returns = sim["net_worth"].pct_change(fill_method=None).dropna()
        if returns.std() > 0:
            sns.kdeplot(returns, label=name, ax=ax, warn_singular=False)
    ax.set_title("Returns distribution (mean trajectory)")
    ax.legend()

    # --- 3. Profit vs Trades (with polyfit guard) ---
    ax = axes[1, 0]
    x, y, labels = [], [], []
    for name, sim in simulations_dict.items():
        profit = sim["net_worth"].iloc[-1] - initial_balance
        trades = num_trades(sim["action"])
        ax.scatter(trades, profit, label=name, zorder=3)
        x.append(trades)
        y.append(profit)
        labels.append(name)

    # only fit if there is actual variance in both axes
    if len(set(x)) > 1 and len(set(y)) > 1 and not any(np.isnan(y)):
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = sorted(x)
            ax.plot(xs, p(xs), linestyle="--", alpha=0.5, color="gray")
        except (np.linalg.LinAlgError, RuntimeWarning):
            pass   # silently skip if polyfit still fails

    ax.set_title("Profit vs trades")
    ax.set_xlabel("Trades")
    ax.set_ylabel("Profit ($)")
    ax.legend()

    # --- 4. MC boxplot ---
    ax = axes[1, 1]
    data, labels = [], []
    for name, runs in mc_dict.items():
        final_vals = [
            pd.to_numeric(r["net_worth"], errors="coerce").ffill().iloc[-1]
            for r in runs
        ]
        data.append(final_vals)
        labels.append(name)
    ax.boxplot(data, tick_labels=labels)
    ax.axhline(initial_balance, linestyle="--", color="gray", alpha=0.5, label="Initial balance")
    ax.set_title("Monte Carlo final wealth distribution")
    ax.legend()

    plt.suptitle(f"{symbol} — secondary analysis dashboard")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{symbol}_secondary_dashboard.png", dpi=150)
    plt.show()

plt.figure(figsize=(8,5))

sharpe_values = [
    calculate_sharpe(
        pd.to_numeric(r["net_worth"], errors="coerce").ffill(),
        freq="1d"
    )
    for r in mc_rl_ens_sent
]

sns.histplot(sharpe_values, kde=True)

plt.title("Sharpe distribution (RL+Ens+Sent)")
plt.show()


plot_main_results(test_df, simulations_dict, simulation_df_bh, initial_balance, symbol)
plot_secondary_dashboard(simulations_dict, mc_dict, initial_balance, symbol)
