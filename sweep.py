"""
sweep.py — One-at-a-time hyperparameter sweep for the RL trading system.
 
Usage:
    python sweep.py --param window_size --values 5 10 20 30
    python sweep.py --param learning_rate --values 0.00005 0.0001 0.0003
    python sweep.py --param lstm_hidden_size --values 32 64 128
    python sweep.py --param xgb_max_depth --values 3 5 7
    python sweep.py --param lambda_efficiency --values 0.0 0.001 0.005 0.01
 
All results are appended to results/sweep_results.csv so you can
compare across runs and across sessions.
"""
 
import argparse
import os
import csv
import yaml
import copy
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
from itertools import product
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
 
# ── your project imports ────────────────────────────────────────────────────
from src.trading_env_global import TradingEnvGlobal
from src.buy_and_hold import buy_and_hold
from evaluation.evaluation_metrics import (
    calculate_sharpe, calculate_max_drawdown, volatility,
    num_trades, total_returns, win_rate, calmar_ratio, annualized_return
)
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
 
# ── paths ───────────────────────────────────────────────────────────────────
BASE_CONFIG   = "configs/config.yaml"
RESULTS_CSV   = "results/sweep_results.csv"
SWEEP_MODELS  = "models/sweep"          # temporary models saved here
N_MC          = 15              # fewer MC runs during sweep for speed
TIMESTEPS     = 10_000                # shorter training during sweep
INITIAL_BAL   = 10_000.0
 
# ── parameter routing ────────────────────────────────────────────────────────
# Maps CLI param name → where in config it lives and what type it is
PARAM_MAP = {
    # Environment
    "window_size":          ("env",      "window_size",                    int),
    "lambda_efficiency":    ("env",      "lambda_efficiency",              float),
    "commission":           ("env",      "commission",                     float),
    # PPO
    "learning_rate":        ("ppo",      "learning_rate",                  float),
    "clip_range":           ("ppo",      "clip_range",                     float),
    "ent_coef":             ("ppo",      "ent_coef",                       float),
    "n_steps":              ("ppo",      "n_steps",                        int),
    "batch_size":           ("ppo",      "batch_size",                     int),
    # XGBoost
    "xgb_horizon":          ("xgboost",  "prediction_horizon_hours",       int),
    "xgb_n_lags":           ("xgboost",  "n_lags",                         int),
    "xgb_max_depth":        ("xgboost",  "params.max_depth",               int),
    "xgb_learning_rate":    ("xgboost",  "params.learning_rate",           float),
    "xgb_n_estimators":     ("xgboost",  "params.n_estimators",            int),
    "xgb_subsample":        ("xgboost",  "params.subsample",               float),
    "xgb_threshold":        ("xgboost",  "threshold",                      float),
    # LSTM
    "lstm_seq_len":         ("lstm",     "sequence_length",                int),
    "lstm_hidden_size":     ("lstm",     "hidden_size",                    int),
    "lstm_epochs":          ("lstm",     "epochs",                         int),
    "lstm_learning_rate":   ("lstm",     "learning_rate",                  float),
    "lstm_horizon":         ("lstm",     "horizon",                        int),
    # Ensemble
    "min_pos_threshold":    ("ensemble", "weighting.min_position_threshold", float),
    "momentum_lb":          ("ensemble", "momentum.lb",                    int),
    "vol_target":           ("ensemble", "volatility_targeting.target_vol", float),
    "max_exposure" :          ("ensemble", "volatility_targeting.target_vol", float),
    "w_lstm":     ("ensemble", "weighting.w_lstm", float),
    "w_xgb":      ("ensemble", "weighting.w_xgb", float),
    "w_momentum": ("ensemble", "weighting.w_momentum", float),
    "ensemble_weights":  ("ensemble", "weighting", dict),
    "vol_lookback":  ("ensemble", "volatility_targeting.vol_lookback", float),
}
 
 
def load_config(path=BASE_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)
 
 
def set_nested(d, dotted_key, value):
    """Set a value in a nested dict using dot notation, e.g. 'params.max_depth'."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
 
 
def apply_param(config, param_name, value):
    """Apply a single parameter value to the right place in config."""
    if param_name not in PARAM_MAP:
        raise ValueError(f"Unknown param: {param_name}. Add it to PARAM_MAP.")
    section, key, dtype = PARAM_MAP[param_name]
    value = dtype(value)
 
    if section == "env":
        config[key] = value
    elif section == "ppo":
        config.setdefault("ppo", {})[key] = value
    elif section == "xgboost":
        set_nested(config.setdefault("ensemble", {}).setdefault("xgboost", {}), key, value)
    elif section == "lstm":
        set_nested(config.setdefault("ensemble", {}).setdefault("lstm", {}), key, value)
    elif section == "ensemble":
        set_nested(config.setdefault("ensemble", {}), key, value)
    return config
 
 
def load_data(config):
    """Load and preprocess the dataframe exactly as in train_model.py."""
    from src.trading_env_global import TradingEnvGlobal
    from evaluation.agent_metrics import (
        prob_up, prob_max_drawdown, signal_entropy,
        macd, relative_strength, ddi, rolling_volatility
    )
    symbol       = config["stock_symbol"]
    processed_dir = config["processed_dir"]
    ensemble_csv  = os.path.join(processed_dir, f"{symbol}_hybrid_ready.csv")
    processed_csv = os.path.join(processed_dir,
        f"{symbol}_sentiment_{config.get('sentiment_source','finnhub_orig')}.csv")
    raw_csv = os.path.join(config["raw_dir"], f"{symbol}_raw.csv")
 
    for path in [ensemble_csv, processed_csv, raw_csv]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
 
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.asfreq("B").ffill().fillna(0)
 
    start = pd.to_datetime(config["start_date"])
    end   = pd.to_datetime(config["end_date"]) if config.get("end_date") else None
    df    = df[df.index >= start] if end is None else df[(df.index >= start) & (df.index <= end)]
    df    = df.sort_index()
    df.name = symbol
 
    df["prob_up"]            = prob_up(df["close"], horizon=1)
    df["prob_max_drawdown"]  = prob_max_drawdown(df["close"], horizon=1, threshold=0.1)
    df["signal_entropy"]     = signal_entropy(df["close"], horizon=1)
    df["macd"]               = macd(df["close"])
    df["rsi"]                = relative_strength(df["close"])
    df["ddi"]                = ddi(df["high"], df["low"], df["close"])
    df["rolling_vol"]        = rolling_volatility(df["close"])
    for col in ["prob_up","prob_max_drawdown","signal_entropy","macd","rsi","ddi","rolling_vol"]:
        df[col] = df[col].fillna(0.0)
 
    split = int(len(df) * 0.8)
    return df.iloc[:split].copy(), df.iloc[split:].copy()
 
 
def run_mc(model, test_df, env_kwargs, n_runs=N_MC, initial_balance=INITIAL_BAL):
    """Run Monte Carlo evaluation and return list of sim DataFrames."""
    results = []
    for _ in range(n_runs):
        env    = TradingEnvGlobal(test_df, **env_kwargs)
        obs, _ = env.reset()
        sim    = pd.DataFrame(index=test_df.index, columns=["net_worth","action"])
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=False )
            action_val = int(action.item())
            obs, _, done, truncated, info = env.step(action_val)
            date = test_df.index[env.step_idx - 1]
            sim.loc[date, "net_worth"] = float(info["net_worth"])
            sim.loc[date, "action"]    = action_val
            if truncated:
                break
        sim["net_worth"] = sim["net_worth"].replace(0, np.nan)
        sim = sim.infer_objects(copy=False)
        sim["action"] = pd.to_numeric(sim["action"], errors="coerce").fillna(0).astype(int)
        results.append(sim)
    return results
 
 
def mc_metrics(mc_runs, test_df, initial_balance=INITIAL_BAL):
    """Compute metrics per MC run, then average metrics."""
    
    ref = test_df.index

    sharpe_vals = []
    final_vals = []
    mdd_vals = []
    vol_vals = []
    total_return_vals = []
    annual_return_vals = []
    calmar_vals = []
    trade_vals = []
    winrate_vals = []

    for r in mc_runs:

        nw = pd.to_numeric(
            r.reindex(ref)["net_worth"],
            errors="coerce"
        )
        nw = nw.dropna()
        acts = pd.to_numeric(
            r.reindex(ref)["action"],
            errors="coerce"
        ).fillna(0).astype(int)

        sharpe_vals.append(
        calculate_sharpe(nw)
    ) 
        rets = nw.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
        final_vals.append(
            nw.iloc[-1]
        )

        mdd_vals.append(
            calculate_max_drawdown(nw)
        )

        vol_vals.append(
            volatility(rets)
        )

        total_return_vals.append(
            total_returns(nw)
        )

        annual_return_vals.append(
            annualized_return(nw)
        )

        calmar_vals.append(
            calmar_ratio(nw)
        )

        trade_vals.append(
            num_trades(acts)
        )

        winrate_vals.append(
            win_rate(nw, acts)
        )
        
    return {
        "sharpe_mean":   np.nanmean(sharpe_vals),
    
        "nw_mean":       np.nanmean(final_vals),


        "total_return":  np.nanmean(total_return_vals),
        "annual_return": np.nanmean(annual_return_vals),

        "num_trades":    np.nanmean(trade_vals),
        "win_rate":      np.nanmean(winrate_vals),
    }
def append_result(row: dict):
    """Append one result row to the sweep CSV."""
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Saved result: {row}")
 
 
def train_and_eval(config, param_name, value, train_df, test_df):
    """Train a PPO model with the given config and evaluate it."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
 
    w   = config.get("window_size", 10)
    com = config.get("commission", 0.0015)
    lam = config.get("lambda_efficiency", 0.0)
 
    env_kwargs = {"use_sentiment": False, "use_ensemble": False,
                  "window_size": w, "commission": com, "lambda_efficiency": lam}
 
    vec_env = make_vec_env(
        lambda: TradingEnvGlobal(train_df, **env_kwargs), n_envs=1
    )
 
    ppo_cfg = config.get("ppo", {})
    model = PPO(
        "MlpPolicy", vec_env, verbose=0, device=device,
        learning_rate = ppo_cfg.get("learning_rate", 0.0001),
        clip_range    = ppo_cfg.get("clip_range", 0.2),
        ent_coef      = ppo_cfg.get("ent_coef", 0.01),
        n_steps       = ppo_cfg.get("n_steps", 2048),
        batch_size    = ppo_cfg.get("batch_size", 64),
    )
    model.learn(total_timesteps=TIMESTEPS)
 
    # save temporarily
    os.makedirs(SWEEP_MODELS, exist_ok=True)
    safe_val = str(value).replace(".", "_")
    model_path = os.path.join(SWEEP_MODELS, f"{param_name}_{safe_val}")
    model.save(model_path)
 
    mc_runs = run_mc(model, test_df, env_kwargs)
    metrics = mc_metrics(mc_runs, test_df)
 
    return metrics
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param",  required=True, help="Parameter name from PARAM_MAP")
    parser.add_argument("--values", required=True, nargs="+", help="Values to test")
    parser.add_argument("--config", default=BASE_CONFIG)
    args = parser.parse_args()
 
    base_config = load_config(args.config)
    train_df, test_df = load_data(base_config)
 
    logger.info(f"Sweeping: {args.param} over values {args.values}")
    logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    all_rows = []
    for val in args.values:

        if args.param == "ensemble_weights":



            weight_values = [float(v) for v in args.values]

            combos = product(weight_values, repeat=3)

            for w_lstm, w_xgb, w_momentum in combos:

                total = w_lstm + w_xgb + w_momentum

                if total == 0:
                    continue

                # normalize
                w_lstm /= total
                w_xgb /= total
                w_momentum /= total

                logger.info(
                    f"\n{'='*50}\n"
                    f"Testing weights:"
                    f" LSTM={w_lstm:.2f}"
                    f" XGB={w_xgb:.2f}"
                    f" MOM={w_momentum:.2f}\n"
                    f"{'='*50}"
                )

                cfg = copy.deepcopy(base_config)

                set_nested(
                    cfg.setdefault("ensemble", {}),
                    "weighting.w_lstm",
                    w_lstm
                )

                set_nested(
                    cfg.setdefault("ensemble", {}),
                    "weighting.w_xgb",
                    w_xgb
                )

                set_nested(
                    cfg.setdefault("ensemble", {}),
                    "weighting.w_momentum",
                    w_momentum
                )

                try:
                    metrics = train_and_eval(
                        cfg,
                        "ensemble_weights",
                        f"{w_lstm:.2f}_{w_xgb:.2f}_{w_momentum:.2f}",
                        train_df,
                        test_df
                    )

                except Exception as e:

                    logger.error(
                        f"Run failed for weights: {e}"
                    )

                    metrics = {
                        k: np.nan for k in [
                            "sharpe_mean",
                            "nw_mean",
                            "max_drawdown",
                            "volatility",
                            "total_return",
                            "num_trades",
                            "win_rate"
                        ]
                    }

                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "param": "ensemble_weights",
                    "value": (
                        f"LSTM={w_lstm:.2f},"
                        f"XGB={w_xgb:.2f},"
                        f"MOM={w_momentum:.2f}"
                    ),
                    "n_mc": N_MC,
                    "timesteps": TIMESTEPS,
                    **metrics
                }

                all_rows.append(row)

        else:
            logger.info(f"\n{'='*50}\nTesting {args.param} = {val}\n{'='*50}")
    
            # deep copy so each run is independent
            cfg = copy.deepcopy(base_config)
            cfg = apply_param(cfg, args.param, val)
    
            try:
                metrics = train_and_eval(cfg, args.param, val, train_df, test_df)
            except Exception as e:
                logger.error(f"Run failed for {args.param}={val}: {e}")
                metrics = {k: np.nan for k in [
                    "sharpe_mean",
                    "nw_mean",
                    "total_return",
                    "num_trades","win_rate"
                ]}
    
            row = {
                "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
                "param":       args.param,
                "value":       val,
                "n_mc":        N_MC,
                "timesteps":   TIMESTEPS,
                **metrics
            }
            all_rows.append(row)
            
    # print summary for this sweep
    df = pd.DataFrame(all_rows)
    df.to_csv(
        RESULTS_CSV,
        index=False,
        float_format="%.6f",
        sep=";",
        decimal=","
    )
    this_sweep = df[df["param"] == args.param]
    print(f"\n{'='*60}")
    print(f"SWEEP RESULTS: {args.param}")
    print(f"{'='*60}")
    print(this_sweep[["value","sharpe_mean","nw_mean",
                       "num_trades","win_rate"]].to_string(index=False))
    best = this_sweep.loc[this_sweep["sharpe_mean"].idxmax()]
    print(f"\nBest value: {best['value']} (Sharpe = {best['sharpe_mean']:.4f})")
 
 
if __name__ == "__main__":
    main()