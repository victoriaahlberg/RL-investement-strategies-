import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import sys
from src.metrics import sharpe_ratio, max_drawdown, annualized_return
import numpy as np

#aquí tenemso ya df_returns, por eso inlcuimos aquí el resumen de las métricas finales 

# Añadir carpetas al path de Python para evitar errores de importación
sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "src", "ensemble"))

from src.ensemble.ensemble_model import EnsembleModel

def run_full_experiment():
    # 1. Cargar configuración
    # Ajusta esta ruta si tu carpeta es 'ensemble/configs' o 'configs'
    config_path = "ensemble/configs/configs_ensemble.yaml"
    if not os.path.exists(config_path):
        config_path = "configs/configs_ensemble.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Descargar datos de NVDA
    print("\n[1/5] Descargando datos de NVDA...")
    df_raw = yf.download("NVDA", start="2022-01-01", end="2024-01-01", interval="1d")

    # Guardar una muestra de df_raw para depuración
    df_raw.head(200).to_csv("debug_df_raw.csv", index=False)
    
    
    print("df_raw guardado en debug_df_raw.csv (primeros 200 registros)")
    
    # Limpieza de columnas de yfinance
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df_raw = df_raw.rename(columns={
        "Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"
    })
    df_raw.index.name = "Date"
    df_raw = df_raw.reset_index()

    # 3. Inicializar el Ensemble
    ensemble = EnsembleModel(config)

    # 4. Dividir Datos (70% Train, 30% Test)
    train_fraction = 0.7 
    split_idx = int(len(df_raw) * train_fraction)
    df_train = df_raw.iloc[:split_idx].copy()
    df_test = df_raw.iloc[split_idx:].copy()

    print(f"\n[2/5] Periodo Entrenamiento: {df_train['Date'].iloc[0].date()} al {df_train['Date'].iloc[-1].date()}")
    print(f"[3/5] Periodo Test (Ciego):  {df_test['Date'].iloc[0].date()} al {df_test['Date'].iloc[-1].date()}")

    # 5. FASE DE ENTRENAMIENTO (Genera los archivos en models/)
    print("\n[4/5] ENTRENANDO MODELOS (XGBoost/LSTM)...")
    # Esto guarda los modelos (.joblib, .pth) usando los datos de entrenamiento
    ensemble.fit_predict(df_train) 

    # 6. FASE DE TEST REALISTA (Out-of-Sample)
    print("\n[5/5] EJECUTANDO PREDICCIÓN CIEGA (Sin trampa)...")
    # predict_out_of_sample cargará los archivos generados arriba
    df_results = ensemble.predict_out_of_sample(df_test)
    #check
    df_results[['Date', 'signal_momentum', 'signal_xgboost', 'signal_lstm']].to_csv('debug_signals.csv', index=False)
    print("Señales guardadas en debug_signals.csv")
    df_results['avg_signal'] = df_results[['signal_momentum', 'signal_xgboost', 'signal_lstm']].mean(axis=1)
   

   # 7. Cálculo de Rendimiento y Visualización
    print("\n[6/6] Generando métricas y gráficas finales...")

    # --- CÁLCULOS ESTADÍSTICOS ---
    df_results['returns'] = df_results['close'].pct_change()
    # La posición de ayer determina el retorno de hoy
    df_results['strat_returns'] = df_results['position'].shift(1) * df_results['returns']
    
    df_results['cum_buy_hold'] = (1 + df_results['returns'].fillna(0)).cumprod()
    df_results['cum_strategy'] = (1 + df_results['strat_returns'].fillna(0)).cumprod() #mide el crecimiento 

    df_results[['Date', 'avg_signal']].to_csv('debug_avg_signal.csv', index=False)
    df_results[['Date', 'cum_strategy', 'cum_buy_hold']].to_csv('debug_cum_returns.csv', index=False)

    initial_balance = 10000

    df_results['net_worth_strategy'] = df_results['cum_strategy'] * initial_balance
    df_results['net_worth_bh'] = df_results['cum_buy_hold'] * initial_balance

    df_results['Date'] = pd.to_datetime(df_results['Date'])
        # === FINAL NET WORTH ===
    final_nw_strategy = df_results['net_worth_strategy'].iloc[-1]
    final_nw_bh = df_results['net_worth_bh'].iloc[-1]

    # === RETURNS LIMPIOS ===
    strat_returns = df_results['strat_returns'].dropna()
    bh_returns = df_results['returns'].dropna()

    # === MÉTRICAS ===
    sharpe_strategy = sharpe_ratio(strat_returns)
    sharpe_bh = sharpe_ratio(bh_returns)

    mdd_strategy = max_drawdown(df_results['net_worth_strategy'])
    mdd_bh = max_drawdown(df_results['net_worth_bh'])

    vol_strategy = strat_returns.std() * np.sqrt(252)
    vol_bh = bh_returns.std() * np.sqrt(252)

    return_strategy = annualized_return(df_results['net_worth_strategy'])
    return_bh = annualized_return(df_results['net_worth_bh'])

    metrics_summary = {
        "Metric": [
            "Final Net Worth",
            "Sharpe Ratio",
            "Volatility",
            "Max Drawdown",
            "Annual Return"
        ],
        "Ensemble": [
            final_nw_strategy,
            sharpe_strategy,
            vol_strategy,
            mdd_strategy,
            return_strategy
        ],
        "Buy_Hold": [
            final_nw_bh,
            sharpe_bh,
            vol_bh,
            mdd_bh,
            return_bh
        ]
    }

    metrics_df = pd.DataFrame(metrics_summary)

    print("\n=== Trading Metrics Summary (Ensemble) ===")
    print(metrics_df)

    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv("results/ensemble_metrics.csv", index=False)

    # --- INICIO DE GRÁFICAS ---
    fig, axes = plt.subplots(4, 1, figsize=(15, 18), sharex=True)

    # Panel 1: Precio + Señales de Ejecución (Flechas)
    axes[0].plot(df_results["Date"], df_results["close"], label="Precio NVDA", alpha=0.5, color='gray')
    
   # Definir umbral para considerar compra o venta
    buy_threshold = 0.5
    sell_threshold = 0.5  # o podrías usar 0 si es long-only

    # En tu script de gráficas: 
    df_results['buy_signal'] = (df_results['position'] > 0) & (df_results['position'].shift(1) <= 0)
    df_results['sell_signal'] = (df_results['position'] < 0) & (df_results['position'].shift(1) >= 0)
    axes[0].plot(df_results["Date"], df_results["close"], label="Precio NVDA", alpha=0.5, color='gray')

    # Usar los booleanos para scatter
    axes[0].scatter(df_results.loc[df_results['buy_signal'], "Date"], 
                df_results.loc[df_results['buy_signal'], "close"], 
                marker="^", color="green", s=100, label="BUY")

    axes[0].scatter(df_results.loc[df_results['sell_signal'], "Date"], 
                df_results.loc[df_results['sell_signal'], "close"], 
                marker="v", color="red", s=100, label="SELL")
    
    total_buys = df_results['buy_signal'].sum()
    total_sells = df_results['sell_signal'].sum()
    print(f"Total BUYs: {total_buys}, Total SELLs: {total_sells}")



    # Panel 2: Señales de los Modelos Individuales
    for col, label, color in [
        ("signal_momentum", "Momentum", "blue"),
        ("signal_xgboost", "XGBoost", "green"),
        ("signal_lstm", "LSTM", "orange")
    ]:
        if col in df_results.columns:
            axes[1].plot(df_results["Date"], df_results[col], label=label, alpha=0.7, color=color)
    
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title("Opinión de los Modelos (Signals)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Exposición Final (El "Grifo" del RL)
    axes[2].plot(df_results["Date"], df_results["position"], label="Exposición (Position)", color="red", linewidth=2)
    axes[2].fill_between(df_results["Date"], 0, df_results["position"], color="red", alpha=0.1)
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].set_title("Exposición Final en Mercado (Controlada por RL)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Rendimiento Acumulado (Dinero)
    axes[3].plot(df_results["Date"], df_results['cum_buy_hold'], label="Buy & Hold (Referencia)", color="gray", linestyle="--")
    axes[3].plot(df_results["Date"], df_results['cum_strategy'], label="Estrategia Ensemble", color="green", linewidth=2)
    axes[3].fill_between(df_results["Date"], 1, df_results['cum_strategy'], where=(df_results['cum_strategy'] >= 1), color='green', alpha=0.1)
    axes[3].fill_between(df_results["Date"], 1, df_results['cum_strategy'], where=(df_results['cum_strategy'] < 1), color='red', alpha=0.1)
    axes[3].set_title("Crecimiento del Capital (Multiplicador)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print("\nProceso finalizado. Analiza el Panel 4 para ver si bates al mercado.")
    print(((df_results['position'] > 0) & (df_results['position'].shift(1) <= 0)).sum())
    print(((df_results['position'] < 0) & (df_results['position'].shift(1) >= 0)).sum())
if __name__ == "__main__":
    run_full_experiment()