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



# === DATA PATH ===
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

      

        while not done:

            action, _ = model.predict(obs, deterministic=False)
            action_val = int(action.item())

            obs, reward, done, truncated, info = env.step(action_val)
            if t >= len(test_df):
                break

            date = test_df.index[env.step_idx - 1]
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
    avg_trade_vals = []
    pn_ratio_vals = []

    for r in mc_runs:

        nw = r["net_worth"].astype(float).ffill()

        action_series = (
            pd.to_numeric(r["action"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # -----------------------------
        # Average return per trade
        # -----------------------------
        n_t = num_trades(action_series)

        total_ret = total_returns(nw)

        avg_return_per_trade = (
            total_ret / n_t if n_t > 0 else 0.0
        )

        avg_trade_vals.append(avg_return_per_trade)

        # -----------------------------
        # Positive / Negative trade ratio
        # -----------------------------
        daily_rets = nw.pct_change(fill_method=None).fillna(0)

        trade_returns = daily_rets[action_series != 0]

        pos = (trade_returns > 0).sum()
        neg = (trade_returns < 0).sum()

        pn_ratio_vals.append(
            pos / neg if neg > 0 else np.nan
        )

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

        # In mc_metrics_summary, inside the for loop:
       "avg_return_per_trade_mean": np.nanmean(avg_trade_vals),
        "avg_return_per_trade_std": np.nanstd(avg_trade_vals),

        "positive_negative_trade_ratio_mean": np.nanmean(pn_ratio_vals),
        "positive_negative_trade_ratio_std": np.nanstd(pn_ratio_vals),
}
    


 
 
results = {name: mc_metrics_summary(runs) for name, runs in mc_dict.items()}
# ============================================
# EXPORT METRICS PER MODEL (Excel-friendly)
# ============================================
bh_final   = simulation_df_bh["net_worth"].iloc[-1]
bh_returns = simulation_df_bh["net_worth"].pct_change(fill_method=None).dropna()  # DEPRECATION FIX
os.makedirs("results/metrics", exist_ok=True)

buy_hold_metrics = {
    "Final Net Worth": bh_final,
    "Sharpe": calculate_sharpe(simulation_df_bh["net_worth"], freq="1d"),
    "Volatility": volatility(bh_returns),
    "Max Drawdown": calculate_max_drawdown(simulation_df_bh["net_worth"]),
    "Total Return": (bh_final / simulation_df_bh["net_worth"].iloc[0]) - 1,
    "Annual Return": annualized_return(simulation_df_bh["net_worth"]),
    "Calmar": np.nan,
    "Num Trades": 1,
    "Win Rate": np.nan,
    "Avg Return Per Trade": np.nan,
    "Pos/Neg Trade Ratio": np.nan,
}

# --------------------------------------------
# Export RL models separately
# --------------------------------------------

metric_mapping = {
    "Final Net Worth": "final_net_worth_mean",
    "Sharpe": "sharpe_mean",
    "Volatility": "volatility_mean",
    "Max Drawdown": "max_drawdown_mean",
    "Total Return": "total_return_mean",
    "Annual Return": "annual_return_mean",
    "Calmar": "calmar_mean",
    "Num Trades": "num_trades_mean",
    "Win Rate": "win_rate_mean",
    "Avg Return Per Trade": "avg_return_per_trade_mean",
    "Pos/Neg Trade Ratio": "positive_negative_trade_ratio_mean"
}

for model_name in ["RL", "RL+Sent", "RL+Ens", "RL+Ens+Sent"]:

    rows = []

    for display_name, metric_key in metric_mapping.items():

        rows.append({
            "Metric": display_name,
            "Value": results[model_name][metric_key]
        })

    # add buy & hold as comparison
    for metric_name, value in buy_hold_metrics.items():

        rows.append({
            "Metric": f"{metric_name} (Buy & Hold)",
            "Value": value
        })

    df_export = pd.DataFrame(rows)

    out_path = f"results/metrics/{symbol}_{model_name}_metrics.csv"

    # Excel Spain friendly
    df_export.to_csv(
        out_path,
        index=False,
        sep=";",
        decimal=","
    )

    print(f"Saved: {out_path}")


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
 

 
metrics_table = pd.DataFrame({
    "Metric": [
        "Final Net Worth", "Sharpe", "Volatility", "Max Drawdown",
        "Total Return", "Annual Return", "Calmar", "Num Trades", "Win Rate", "Avg Return Per Trade",
        "Pos/Neg Trade Ratio"
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
            results[name]["avg_return_per_trade_mean"],
            results[name]["positive_negative_trade_ratio_mean"]
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
        np.nan,   # avg return per trade
        np.nan,   # pos/neg ratio  
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
##testnig to see if correct CSV format 
for col in results_df.columns:
    results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
 
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


# Run this once per stock, saving results each time
# Then call the plot function at the end

results_store = {}  # accumulate across stocks

for stock in ["AAPL", "TSLA", "META", "MSFT", "AMZN", 
               "NVDA", "GOOGL", "IDR", "ITX", "SPY"]:
    
    # Change symbol in config and run train_model.py
    # Then save the outputs:
    results_store[stock] = {
        "test_df": test_df,
        "mc_rl_no_sent": mc_rl_no_sent,
        "mc_rl_sent": mc_rl_sent,
        "mc_rl_ens_no_sent": mc_rl_ens_no_sent,
        "mc_rl_ens_sent": mc_rl_ens_sent,
        "bh": simulation_df_bh,
    }
    
    # Save to disk so you don't lose it between runs
    import pickle
    with open(f"results/{stock}_results.pkl", "wb") as f:
        pickle.dump(results_store[stock], f)

def align(sim):
    return sim.reindex(test_df.index).ffill()

simulations_dict = {
        "RL": simulation_rl_no_sent,
        "RL+Sent": simulation_rl_sent,
        "RL+Ens": simulation_rl_ens_no_sent,
        "RL+Ens+Sent": simulation_rl_ens_sent
    }

def plot_main_results(test_df, simulations_dict, simulation_df_bh, initial_balance, symbol):
    models = {
    "RL": simulation_rl_no_sent,
    "RL+Sent": simulation_rl_sent,
    "RL+Ens": simulation_rl_ens_no_sent,
    "RL+Ens+Sent": simulation_rl_ens_sent
    }
    

    # =========================
    # 1. PRECIO + ACCIONES (solo modelo final)
    # =========================
    for name, sim in models.items():

        sim = sim.reindex(test_df.index).ffill()

        buy = sim["action"].fillna(0) == 1
        sell = sim["action"].fillna(0) == 2

        # =====================================
        # 1. BUY / SELL
        # =====================================

        plt.figure(figsize=(14,5))

        plt.plot(
            test_df.index,
            test_df["close"],
            label="Close"
        )

        plt.scatter(
            test_df.index[buy],
            test_df["close"][buy],
            marker="^",
            color="green",
            label="BUY",
            alpha=0.7
        )

        plt.scatter(
            test_df.index[sell],
            test_df["close"][sell],
            marker="v",
            color="red",
            label="SELL",
            alpha=0.7
        )

        plt.title(f"{symbol} — {name} Buy/Sell Signals")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"results/{symbol}_{name}_buy_sell.png",
            dpi=150
        )

        plt.close()

        # =====================================
        # 2. NET WORTH
        # =====================================

        plt.figure(figsize=(14,5))

        plt.plot(
            sim.index,
            sim["net_worth"],
            label=name
        )

        plt.plot(
            simulation_df_bh.index,
            simulation_df_bh["net_worth"],
            "--",
            color="black",
            label="Buy & Hold"
        )

        plt.title(f"{symbol} — {name} Net Worth")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"results/{symbol}_{name}_networth.png",
            dpi=150
        )

        plt.close()

        # =====================================
        # 3. P&L
        # =====================================

        plt.figure(figsize=(14,5))

        pnl_model = sim["net_worth"] - initial_balance
        pnl_bh = simulation_df_bh["net_worth"] - initial_balance

        plt.plot(
            sim.index,
            pnl_model,
            label=name
        )

        plt.plot(
            simulation_df_bh.index,
            pnl_bh,
            "--",
            color="black",
            label="Buy & Hold"
        )

        plt.axhline(
            0,
            linestyle="--",
            color="gray"
        )

        plt.title(f"{symbol} — {name} Profit & Loss")
        plt.ylabel("P&L ($)")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"results/{symbol}_{name}_pnl.png",
            dpi=150
        )

        plt.close()
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
            sns.histplot(
                all_actions,
                discrete=True,
                stat="probability",
                shrink=0.8,
                ax=ax,
                label=name
            )
        else:
            ax.axvline(all_actions[0], label=f"{name} (constant)", linestyle="--")
    ax.set_title("Action distribution (pooled MC runs)")
    ax.set_xlabel("Action  (0=Hold, 1=Buy, 2=Sell)")
    ax.legend(loc="best")
    # --- 2. Returns distribution ---
    # --- 2. Returns distribution ---
    ax = axes[0, 1]

    data = []
    labels = []

    for name, runs in mc_dict.items():

        final_vals = []

        for r in runs:

            vals = pd.to_numeric(
                r["net_worth"],
                errors="coerce"
            ).ffill()

            final_vals.append(vals.iloc[-1])

        data.append(final_vals)
        labels.append(name)

    # violin
    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )

    # transparency
    for pc in parts['bodies']:
        pc.set_alpha(0.4)

    # overlay boxplot
    ax.boxplot(
        data,
        positions=np.arange(1, len(labels)+1),
        widths=0.15
    )

    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=15)

    ax.axhline(
        initial_balance,
        linestyle='--',
        color='gray',
        alpha=0.5
    )

    ax.set_title("Final wealth distribution")
    ax.set_ylabel("Final net worth ($)")

    # --- 3. Profit vs Trades (with polyfit guard) ---
    ax = axes[1, 0]
    names = list(simulations_dict.keys())
    profits = [sim['net_worth'].iloc[-1] - initial_balance for sim in simulations_dict.values()]
    trades = [num_trades(sim['action']) for sim in simulations_dict.values()]

    bars = ax.bar(names, profits, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    for bar, t in zip(bars, trades):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(t)} trades', ha='center', va='bottom', fontsize=9)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Profit by model (trades annotated)")
    ax.set_ylabel("Profit ($)")
    

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

def plot_metrics_barchart(metrics_table, symbol):
    metrics_to_plot = ['Sharpe', 'Total Return', 'Max Drawdown', 'Win Rate']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 5))
    models = [c for c in metrics_table.columns if c != 'Metric']
    colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']

    for ax, metric in zip(axes, metrics_to_plot):
        row = metrics_table[metrics_table['Metric'] == metric]
        if row.empty:
            continue
        vals = [float(row[m].values[0]) if pd.notna(row[m].values[0]) else 0 for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)])
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=30)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.4)

    plt.suptitle(f"{symbol} — Key metrics comparison")
    plt.tight_layout()
    plt.savefig(f"results/{symbol}_metrics_barchart.png", dpi=150)
    plt.show()


plot_main_results(test_df, simulations_dict, simulation_df_bh, initial_balance, symbol)
""""
plot_secondary_dashboard(simulations_dict, mc_dict, initial_balance, symbol)
plot_metrics_barchart(metrics_table, symbol)
"""
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_10_asset_networth(
    symbols,
    test_dfs,
    mc_dicts,
    bh_dicts,
    initial_balance=10000,
    best_config="RL+Ens"
):
    """
    symbols:       list of 10 tickers, AAPL first
    test_dfs:      dict {symbol: test_df}
    mc_dicts:      dict {symbol: {config_name: mc_runs_list}}
    bh_dicts:      dict {symbol: simulation_df_bh}
    best_config:   which config to plot for secondary assets
    """

    def mc_to_mean_series(mc_runs, reference_index, initial_balance):
        aligned = []
        for r in mc_runs:
            nw = pd.to_numeric(
                r.reindex(reference_index)["net_worth"],
                errors="coerce"
            ).ffill().fillna(initial_balance)
            aligned.append(nw.to_numpy())
        arr = np.vstack(aligned)
        return pd.Series(arr.mean(axis=0), index=reference_index)

    # ── Layout: 2 rows × 5 cols ────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

    config_colors = {
        "RL":          "#E53935",
        "RL+Sent":     "#FB8C00",
        "RL+Ens":      "#43A047",
        "RL+Ens+Sent": "#1E88E5",
    }

    for idx, symbol in enumerate(symbols):
        row, col = divmod(idx, 5)
        ax = fig.add_subplot(gs[row, col])

        ref_idx = test_dfs[symbol].index

        # ── AAPL: plot all 4 configurations ──────────────────────────
        if symbol == "AAPL":
            for cfg_name, mc_runs in mc_dicts[symbol].items():
                mean_nw = mc_to_mean_series(mc_runs, ref_idx, initial_balance)
                ax.plot(
                    mean_nw.index, mean_nw.values,
                    label=cfg_name,
                    color=config_colors.get(cfg_name, "gray"),
                    linewidth=1.8
                )
            ax.set_title("AAPL (full ablation)",
                         fontsize=9, fontweight="bold")

        # ── Other 9 assets: best config only ─────────────────────────
        else:
            if best_config in mc_dicts[symbol]:
                mean_nw = mc_to_mean_series(
                    mc_dicts[symbol][best_config], ref_idx, initial_balance
                )
                ax.plot(
                    mean_nw.index, mean_nw.values,
                    label=best_config,
                    color=config_colors.get(best_config, "#1E88E5"),
                    linewidth=1.8
                )
            ax.set_title(symbol, fontsize=9, fontweight="bold")

        # ── Buy & Hold for every asset ────────────────────────────────
        bh = bh_dicts[symbol]
        ax.plot(
            bh.index, bh["net_worth"].values,
            label="Buy & Hold",
            color="black",
            linewidth=1.1,
            linestyle="--",
            alpha=0.7
        )

        # ── Reference line ────────────────────────────────────────────
        ax.axhline(
            initial_balance,
            color="gray", linestyle=":", linewidth=0.8, alpha=0.5
        )

        # ── Formatting ────────────────────────────────────────────────
        ax.set_ylabel("Net worth ($)", fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.25)

        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")
        else:
            # compact legend for secondary assets
            ax.legend(fontsize=6, loc="upper left",
                      labels=[best_config, "Buy & Hold"])

    fig.suptitle(
        f"Net worth trajectories — {best_config} vs Buy & Hold (10 assets)\n"
        f"AAPL shows full ablation study",
        fontsize=12, fontweight="bold", y=1.01
    )

    os.makedirs("results", exist_ok=True)
    plt.savefig(
        "results/multi_asset_10_networth.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Saved: results/multi_asset_10_networth.png")

import pickle
results_store = {}
symbols = [
    "AAPL",
    "TSLA",
    "META",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "IDR",
    "ITX",
    "SPY"
]

results_store = {}
for stock in symbols:
    with open(f"results/{stock}_results.pkl", "rb") as f:
        results_store[stock] = pickle.load(f)

# Then build the dicts the plot function expects
test_dfs = {s: results_store[s]["test_df"] for s in results_store}
bh_dicts = {s: results_store[s]["bh"] for s in results_store}
mc_dicts = {s: {
    "RL":          results_store[s]["mc_rl_no_sent"],
    "RL+Sent":     results_store[s]["mc_rl_sent"],
    "RL+Ens":      results_store[s]["mc_rl_ens_no_sent"],
    "RL+Ens+Sent": results_store[s]["mc_rl_ens_sent"],
} for s in results_store}

plot_10_asset_networth(
    symbols,
    test_dfs,
    mc_dicts,
    bh_dicts
)
import os
import matplotlib.pyplot as plt

def plot_aapl_ablation(simulations_dict, bh_df, test_df, initial_balance):
    """
    simulations_dict must contain:
        - RL
        - RL+Sent
        - RL+Ens
        - RL+Ens+Sent
    """

    os.makedirs("results", exist_ok=True)

    configs = ["RL", "RL+Sent", "RL+Ens", "RL+Ens+Sent"]

    cumulative = {}

    for i, cfg in enumerate(configs):

        cumulative[cfg] = simulations_dict[cfg]

        plt.figure(figsize=(12, 5))

        # --- Plot Buy & Hold ---
        plt.plot(
            bh_df.index,
            bh_df["net_worth"],
            label="Buy & Hold",
            linestyle="--",
            color="black"
        )

        # --- Plot all configs up to current step ---
        for name in configs[:i+1]:
            sim = cumulative[name]
            sim = sim.reindex(test_df.index).ffill()
            plt.plot(
                sim.index,
                sim["net_worth"],
                label=name
            )

        plt.title(f"AAPL Ablation Study — up to {cfg}")
        plt.ylabel("Net Worth ($)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(alpha=0.3)

        path = f"results/AAPL_ablation_{i+1}_{cfg}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"Saved: {path}")
"""
plot_aapl_ablation(
    simulations_dict={
        "RL": simulation_rl_no_sent,
        "RL+Sent": simulation_rl_sent,
        "RL+Ens": simulation_rl_ens_no_sent,
        "RL+Ens+Sent": simulation_rl_ens_sent
    },
    bh_df=simulation_df_bh,
    test_df=test_df,
    initial_balance=10000
)"""