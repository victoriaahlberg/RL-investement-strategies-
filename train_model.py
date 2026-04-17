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


# Initialize environments
vec_env_with_sentiment = make_vec_env(
    lambda: TradingEnv(train_df, use_sentiment=True, use_ensemble=True),
    n_envs=1
)
vec_env_without_sentiment = make_vec_env(
    lambda: TradingEnv(train_df, use_sentiment=False, use_ensemble=False),
    n_envs=1
)

vec_env_rl_base = make_vec_env(
    lambda: TradingEnv(train_df, use_sentiment=True, use_ensemble=False),
    n_envs=1
)

# Set device (MPS or CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- Train PPO model with sentiment ---
# --- CAMBIO 3: Nombres descriptivos ---
model_path_with = "models/ppo_expert_ensemble"   # Con todo
model_path_without = "models/ppo_base_sentiment" # Solo sentimiento

if os.path.exists(model_path_with + ".zip"):
    # Cargar modelo previo y continuar entrenamiento
    model_with_sentiment = PPO.load(model_path_with)
    model_with_sentiment.set_env(vec_env_with_sentiment)
    additional_timesteps = 100_000 
    logger.info(f"Continuing training PPO with sentiment for {additional_timesteps} timesteps")
    model_with_sentiment.learn(total_timesteps=additional_timesteps)
else:

    # Entrenamiento desde cero
    model_with_sentiment = PPO(
        "MlpPolicy",
        vec_env_with_sentiment,
        verbose=1,
        device=device,
        learning_rate=0.0001,
        clip_range=0.2,
        ent_coef=0.01
    )
    logger.info(f"Training PPO with sentiment for {config['timesteps']} timesteps")
    model_with_sentiment.learn(total_timesteps=config['timesteps'])
model_with_sentiment.save(model_path_with)
logger.info(f"Model with sentiment saved to {model_path_with}")

# Train PPO model without sentiment

#caso: ya existe un modelo

# 🚨 IMPORTANTE: no reutilizar modelo si cambia el observation space
if os.path.exists(model_path_without + ".zip"):
    #cargamos el modelo previamente entrenado
    model_without_sentiment = PPO.load(model_path_without)
    #asignamos el entorno actual donde continuara el entrenamiento
    model_without_sentiment.set_env(vec_env_without_sentiment)
    additional_timesteps = 100_000 
    logger.info(f"Continuing training PPO without sentiment for {additional_timesteps} timesteps")
    model_without_sentiment.learn(total_timesteps=additional_timesteps)
else:

    model_without_sentiment = PPO(
        "MlpPolicy",
        vec_env_without_sentiment,
        verbose=1,
        device=device,
        learning_rate=0.0001,
        clip_range=0.2,
        ent_coef=0.01
    )
    logger.info(f"Training PPO without sentiment for {config['timesteps']} timesteps")
    model_without_sentiment.learn(total_timesteps=config['timesteps'])
model_without_sentiment.save(model_path_without)
logger.info(f"Model without sentiment saved to {model_path_without}")

model_path_base = "models/ppo_rl_base"

if os.path.exists(model_path_base + ".zip"):
    model_base = PPO.load(model_path_base)
    model_base.set_env(vec_env_rl_base)
    model_base.learn(total_timesteps=100_000)
else:
    model_base = PPO(
        "MlpPolicy",
        vec_env_rl_base,
        verbose=1,
        device=device,
        learning_rate=0.0001,
        clip_range=0.2,
        ent_coef=0.01
    )
    model_base.learn(total_timesteps=config['timesteps'])

model_base.save(model_path_base)

# Initialize simulation DataFrames with Date as index
simulation_df_with_sentiment = pd.DataFrame(index=test_df.index, columns=['net_worth', 'action'])
simulation_df_without_sentiment = pd.DataFrame(index=test_df.index, columns=['net_worth', 'action'])

# ------------------------
# Buy & Hold baseline
# ------------------------
simulation_df_bh, metrics_bh, actions_bh, final_buy_hold = buy_and_hold(test_df, initial_balance=initial_balance)
logger.info(f"Buy & Hold Metrics: Sharpe={metrics_bh['sharpe']:.2f}, MDD={metrics_bh['max_drawdown']:.1%}, "
            f"Vol={metrics_bh['volatility']:.1%}, Trades={metrics_bh['num_trades']}")



#Aquí ya no entrenamos, evaluamos 
# --- SIMULACIÓN CORREGIDA EN train_model.py ---
env = TradingEnv(test_df, use_sentiment=True, use_ensemble=True)
obs = env.reset()[0]

# El rango es correcto, pero debemos sincronizar los índices
for step in range(len(test_df) - env.window_size):
    action, _ = model_with_sentiment.predict(obs)
    action_val = action.item()
    
    # 1. Ejecutar el paso. El entorno ya calcula el Net Worth internamente de forma correcta.
    obs, reward, done, truncated, info = env.step(action_val)
    
    # 2. RECUPERAR LOS VALORES REALES DEL ENTORNO
    # No calcules el Net Worth fuera; usa el que el entorno ya calculó tras el movimiento del precio.
    net_worth_value = env.net_worth 
    
    # 3. SINCRONIZAR LA FECHA
    # La fecha debe ser la del paso actual del entorno (que ya avanzó en env.step)
    current_idx = env.current_step - 1 
    date = test_df.index[current_idx]
    
    # Guardar en tu DataFrame de resultados
    simulation_df_with_sentiment.loc[date, 'net_worth'] = net_worth_value
    simulation_df_with_sentiment.loc[date, 'action'] = action_val
    
    logger.debug(f"Step {step} | Date {date} | Action {action_val} | Net Worth {net_worth_value:.2f}")

    if done or truncated:
        # Tu lógica de rellenado de fechas faltantes permanece igual
        missing_dates = test_df.index[current_idx + 1:].tolist()
        if missing_dates:
            logger.warning(f"Simulación terminada antes. Rellenando {len(missing_dates)} fechas.")
            for m_date in missing_dates:
                simulation_df_with_sentiment.loc[m_date, 'net_worth'] = net_worth_value
                simulation_df_with_sentiment.loc[m_date, 'action'] = 0 # Hold
        break

# Simulate trading (without sentiment)
# Simulate trading (without sentiment)
env = TradingEnv(test_df, use_sentiment=False, use_ensemble=False)
obs = env.reset()[0]

for step in range(len(test_df) - env.window_size):
    action, _ = model_without_sentiment.predict(obs)
    action_val = action.item()
    
    # 1. Ejecutamos el paso (el entorno actualiza balance, acciones y net_worth)
    obs, reward, done, truncated, info = env.step(action_val)
    
    # 2. SINCRONIZACIÓN CORRECTA:
    # Usamos el paso real en el que está el entorno para buscar la fecha
    current_idx = env.current_step - 1 
    date = test_df.index[current_idx]
    
    # Usamos el Net Worth que el entorno ya calculó internamente (es el más fiable)
    net_worth_value = env.net_worth 
    
    simulation_df_without_sentiment.loc[date, 'net_worth'] = net_worth_value
    simulation_df_without_sentiment.loc[date, 'action'] = action_val
    
    logger.debug(f"Step {step} (without): action={action_val}, net_worth={net_worth_value}, date={date}")
    
    if done or truncated:
        # Rellenar fechas faltantes hasta el final del dataset si se corta antes
        remaining_dates = test_df.index[current_idx + 1:]
        if not remaining_dates.empty:
            simulation_df_without_sentiment.loc[remaining_dates, 'net_worth'] = net_worth_value
            simulation_df_without_sentiment.loc[remaining_dates, 'action'] = 0
        break

simulation_df_base = pd.DataFrame(index=test_df.index, columns=['net_worth', 'action'])

env = TradingEnv(test_df, use_sentiment=True, use_ensemble=False)
obs = env.reset()[0]

for step in range(len(test_df) - env.window_size):
    action, _ = model_base.predict(obs)
    action_val = action.item()

    obs, reward, done, truncated, info = env.step(action_val)

    current_idx = env.current_step - 1
    date = test_df.index[current_idx]
    net_worth_value = env.net_worth

    simulation_df_base.loc[date, 'net_worth'] = net_worth_value
    simulation_df_base.loc[date, 'action'] = action_val

    if done or truncated:
        break

# === PASO FINAL OBLIGATORIO PARA ELIMINAR EL VALUEERROR ===
# Rellenamos los huecos de la ventana inicial (días 0 a 10) que el bucle no toca
simulation_df_without_sentiment['net_worth'] = simulation_df_without_sentiment['net_worth'].fillna(initial_balance)
simulation_df_without_sentiment['action'] = simulation_df_without_sentiment['action'].fillna(0)

# Aseguramos que todo sea float para las métricas
simulation_df_without_sentiment['net_worth'] = simulation_df_without_sentiment['net_worth'].astype(float)

# === RELLENADO DE SEGURIDAD PARA AMBOS MODELOS ===
# Esto elimina los NaNs de los primeros 10 días (window_size)
for sim_df in [simulation_df_with_sentiment, simulation_df_without_sentiment]:
    # 1. Rellenar la ventana inicial (donde el agente no podía operar)
    sim_df['net_worth'] = sim_df['net_worth'].fillna(initial_balance)
    sim_df['action'] = sim_df['action'].fillna(0)
    
    # 2. Por si acaso el entorno terminó antes del final del dataset
    sim_df['net_worth'] = sim_df['net_worth'].ffill()
    sim_df['action'] = sim_df['action'].fillna(0)
    
    # 3. Solución al "FutureWarning" que te salía en consola:
    # Convertimos explícitamente a tipos numéricos para evitar avisos de pandas
    sim_df['net_worth'] = pd.to_numeric(sim_df['net_worth'])
    sim_df['action'] = pd.to_numeric(sim_df['action']).astype(int)

# Ahora las validaciones ya no fallarán
logger.info("Date indices and data aligned successfully")

# Verify alignment and data completeness
if not simulation_df_with_sentiment.index.equals(test_df.index):
    logger.error("Index misalignment in simulation_df_with_sentiment")
    raise ValueError("Index misalignment in simulation_df_with_sentiment")
if not simulation_df_without_sentiment.index.equals(test_df.index):
    logger.error("Index misalignment in simulation_df_without_sentiment")
    raise ValueError("Index misalignment in simulation_df_without_sentiment")
if simulation_df_with_sentiment['net_worth'].isna().any():
    logger.error("Missing net_worth values in simulation_df_with_sentiment")
    raise ValueError("Missing net_worth values in simulation_df_with_sentiment")
if simulation_df_without_sentiment['net_worth'].isna().any():
    logger.error("Missing net_worth values in simulation_df_without_sentiment")
    raise ValueError("Missing net_worth values in simulation_df_without_sentiment")
logger.info("Date indices and data aligned successfully")

sharpe_with_sentiment = calculate_sharpe(simulation_df_with_sentiment['net_worth'])
sharpe_without_sentiment = calculate_sharpe(simulation_df_without_sentiment['net_worth'])

mdd_with_sentiment = calculate_max_drawdown(simulation_df_with_sentiment['net_worth'])
mdd_without_sentiment = calculate_max_drawdown(simulation_df_without_sentiment['net_worth'])

vol_with_sentiment = volatility(simulation_df_with_sentiment['net_worth'])
vol_without_sentiment = volatility(simulation_df_without_sentiment['net_worth'])

trades_with_sentiment = num_trades(simulation_df_with_sentiment['action'])
trades_without_sentiment = num_trades(simulation_df_without_sentiment['action'])


# Predicción para el día siguiente
#inicializar el entorno
# ==============================================================================
# Acción recomendada para mañana
# ==============================================================================
# Inicializamos el entorno incluyendo use_ensemble=True
env = TradingEnv(test_df, use_sentiment=True, use_ensemble=True)
obs = env.reset()[0]

# Avanzar hasta el último día del dataset para obtener la observación final
done = False
while not done:
    obs, _, done, _, _ = env.step(0) # Avanzar sin hacer operaciones (HOLD)

# Llamamos al modelo experto para predecir lo de mañana
action_tomorrow, _ = model_with_sentiment.predict(obs)
action_tomorrow = action_tomorrow.item()
action_map = {0: "HOLD ⚪", 1: "BUY 🟢", 2: "SELL 🔴"}
print(f"\nRecomendación para mañana (Modelo con Ensemble): {action_map[action_tomorrow]}")

# Log de métricas principales
logger.info(f"Sharpe (with ensemble): {sharpe_with_sentiment:.2f}")
logger.info(f"Sharpe (without ensemble): {sharpe_without_sentiment:.2f}")
logger.info(f"Max Drawdown (with ensemble): {mdd_with_sentiment:.2%}")
logger.info(f"Max Drawdown (without ensemble): {mdd_without_sentiment:.2%}")
logger.info(f"Volatility (with ensemble): {vol_with_sentiment:.2%}")
logger.info(f"Volatility (without ensemble): {vol_without_sentiment:.2%}")
logger.info(f"Number of trades (with ensemble): {trades_with_sentiment}")
logger.info(f"Number of trades (without ensemble): {trades_without_sentiment}")

# Save results
results_df = pd.DataFrame({
    'Date': test_df.index,
    'Net_Worth_With_Ensemble': simulation_df_with_sentiment['net_worth'],
    'Net_Worth_Without_Ensemble': simulation_df_without_sentiment['net_worth'],
    'Actions_With_Ensemble': simulation_df_with_sentiment['action'],
    'Actions_Without_Ensemble': simulation_df_without_sentiment['action']
})

# Guardar resultados en CSV
os.makedirs('results', exist_ok=True)
results_df['Net_Worth_Buy_Hold'] = simulation_df_bh['net_worth']
results_df.to_csv('results/aapl_trading_results.csv', index=False)
logger.info("Saved trading results to results/aapl_trading_results.csv")

# ==============================================================================
# Generate comparison plots (Manteniendo tu estructura original)
# ==============================================================================
def plot_results(
    test_df,
    net_worth_with_sentiment,
    actions_with_sentiment,
    net_worth_without_sentiment,
    actions_without_sentiment,
    sharpe_with_sentiment,
    sharpe_without_sentiment,
    mdd_with_sentiment,
    mdd_without_sentiment,
    vol_with_sentiment,
    vol_without_sentiment,
    trades_with_sentiment,
    trades_without_sentiment,
    symbol
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 1. Precio y acciones (Usando el modelo con Ensemble)
    test_df_plot = test_df.iloc[-len(net_worth_with_sentiment):]
    ax1.plot(test_df_plot.index, test_df_plot['close'], label='Close Price', color='blue', alpha=0.7)
    buy_points = test_df_plot[actions_with_sentiment.values == 1]
    sell_points = test_df_plot[actions_with_sentiment.values == 2]
    ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', label='Buy (With Ensemble)')
    ax1.scatter(sell_points.index, sell_points['close'], color='red', marker='v', label='Sell (With Ensemble)')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} Close Price & Trading Actions (Hybrid RL)')
    ax1.legend(loc='upper left')

    # 2. Net Worth Comparison
    ax2.plot(test_df_plot.index, simulation_df_bh['net_worth'],
             label=f'Buy & Hold\nSharpe: {metrics_bh["sharpe"]:.2f}, MDD: {metrics_bh["max_drawdown"]:.1%}',
             color='green', linestyle=':')
    
    ax2.plot(test_df_plot.index, net_worth_with_sentiment,
             label=f'With Ensemble\nSharpe: {sharpe_with_sentiment:.2f}, MDD: {mdd_with_sentiment:.1%}, Vol: {vol_with_sentiment:.1%}',
             color='purple', linewidth=2)
    
    ax2.plot(test_df_plot.index, net_worth_without_sentiment,
             label=f'Without Ensemble\nSharpe: {sharpe_without_sentiment:.2f}, MDD: {mdd_without_sentiment:.1%}, Vol: {vol_without_sentiment:.1%}',
             color='orange', linewidth=2, linestyle='--')
    
    ax2.set_ylabel('Net Worth ($)')
    ax2.set_xlabel('Date')
    ax2.set_title('Portfolio Net Worth Comparison')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plot_path = f"results/{symbol.lower()}_trading_results_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Llamada a la función de gráfico
plot_results(
    df,
    simulation_df_with_sentiment['net_worth'],
    simulation_df_with_sentiment['action'],
    simulation_df_without_sentiment['net_worth'],
    simulation_df_without_sentiment['action'],
    sharpe_with_sentiment,
    sharpe_without_sentiment,
    mdd_with_sentiment,
    mdd_without_sentiment,
    vol_with_sentiment,
    vol_without_sentiment,
    trades_with_sentiment,
    trades_without_sentiment,
    symbol
)

# ==============================================================================
# Resumen Final de Métricas (metrics_summary)
# ==============================================================================
returns_with = simulation_df_with_sentiment['net_worth'].pct_change().dropna() #ensemble + sent
returns_without = simulation_df_without_sentiment['net_worth'].pct_change().dropna() #sin 
returns_bh = simulation_df_bh['net_worth'].pct_change().dropna()
returns_base = simulation_df_base['net_worth'].pct_change().dropna() #sin ensemble con sent

metrics_summary = {
    "Metric": [
        "Final Net Worth", "Sharpe Ratio", "Volatility", "Max Drawdown",
        "Total Return", "Win Rate", "Calmar Ratio", "Number of trades", "Annualized returns"
    ],
    "With_Ensemble": [
        simulation_df_with_sentiment['net_worth'].iloc[-1],
        calculate_sharpe(simulation_df_with_sentiment['net_worth'], freq="1d"),
        volatility(returns_with),
        calculate_max_drawdown(simulation_df_with_sentiment['net_worth']),
        total_returns(simulation_df_with_sentiment['net_worth']),
        win_rate(simulation_df_with_sentiment['net_worth'], simulation_df_with_sentiment['action']),
        calmar_ratio(simulation_df_with_sentiment['net_worth']),
        num_trades(simulation_df_with_sentiment['action']),
        annualized_return(simulation_df_with_sentiment['net_worth'])
    ],
    "Without_Ensemble": [
        simulation_df_without_sentiment['net_worth'].iloc[-1],
        calculate_sharpe(simulation_df_without_sentiment['net_worth'], freq="1d"),
        volatility(returns_without),
        calculate_max_drawdown(simulation_df_without_sentiment['net_worth']),
        total_returns(simulation_df_without_sentiment['net_worth']),
        win_rate(simulation_df_without_sentiment['net_worth'], simulation_df_without_sentiment['action']),
        calmar_ratio(simulation_df_without_sentiment['net_worth']),
        num_trades(simulation_df_without_sentiment['action']),
        annualized_return(simulation_df_without_sentiment['net_worth'])
    ],
    "Buy_Hold": [
        simulation_df_bh['net_worth'].iloc[-1],
        calculate_sharpe(simulation_df_bh['net_worth'], freq="1d"),
        volatility(returns_bh),
        calculate_max_drawdown(simulation_df_bh['net_worth']),
        total_returns(simulation_df_bh['net_worth']),
        win_rate(simulation_df_bh['net_worth'], actions_bh),
        calmar_ratio(simulation_df_bh['net_worth']),
        num_trades(actions_bh),
        annualized_return(simulation_df_bh['net_worth'])
        ],
    "Base model":[
        simulation_df_base['net_worth'].iloc[-1],
        calculate_sharpe(simulation_df_base['net_worth'], freq="1d"),
        volatility(returns_base),
        calculate_max_drawdown(simulation_df_base['net_worth']),
        total_returns(simulation_df_base['net_worth']),
        win_rate(simulation_df_base['net_worth'], simulation_df_base['action']),
        calmar_ratio(simulation_df_base['net_worth']),
        num_trades(simulation_df_base['action']),
        annualized_return(simulation_df_base['net_worth'])
        ]
    }

metrics_df = pd.DataFrame(metrics_summary)
print("\n=== Trading Metrics Summary ===")
print(metrics_df.to_string(index=False))

metrics_csv_path = "results/trading_metrics_summary.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
logger.info(f"Saved metrics summary to {metrics_csv_path}")