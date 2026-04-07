import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import sys

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

   # 7. Cálculo de Rendimiento y Visualización
    print("\n[6/6] Generando métricas y gráficas finales...")

    # --- CÁLCULOS ESTADÍSTICOS ---
    df_results['returns'] = df_results['close'].pct_change()
    # La posición de ayer determina el retorno de hoy
    df_results['strat_returns'] = df_results['position'].shift(1) * df_results['returns']
    
    df_results['cum_buy_hold'] = (1 + df_results['returns'].fillna(0)).cumprod()
    df_results['cum_strategy'] = (1 + df_results['strat_returns'].fillna(0)).cumprod()

    # --- INICIO DE GRÁFICAS ---
    fig, axes = plt.subplots(4, 1, figsize=(15, 18), sharex=True)

    # Panel 1: Precio + Señales de Ejecución (Flechas)
    axes[0].plot(df_results["Date"], df_results["close"], label="Precio NVDA", alpha=0.5, color='gray')
    
    # Lógica de flechas: buscamos cambios de dirección en la posición
    compras = df_results[(df_results['position'].shift(1) <= 0) & (df_results['position'] > 0)]
    ventas = df_results[(df_results['position'].shift(1) > 0) & (df_results['position'] <= 0)]
    
    axes[0].scatter(compras["Date"], compras["close"], marker="^", color="green", s=100, label="BUY (Entrada)", zorder=5)
    axes[0].scatter(ventas["Date"], ventas["close"], marker="v", color="red", s=100, label="SELL (Salida)", zorder=5)
    axes[0].set_title("Precio de NVDA y Ejecución de Trades")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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

if __name__ == "__main__":
    run_full_experiment()