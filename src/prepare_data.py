import pandas as pd
import yaml
import os
import sys
import yfinance as yf

# Configuración de rutas para encontrar tus módulos
sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "src", "ensemble"))

from src.ensemble.ensemble_model import EnsembleModel
from src.gen_utils import load_config

def generate_super_csv():
    # 1. Cargar la configuración
    config_path = "configs/configs_ensemble.yaml"
    if not os.path.exists(config_path):
        config_path = "ensemble/configs/configs_ensemble.yaml"
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg.get("stock_symbol")
    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")
    interval = cfg.get("data_interval")

    if not symbol or not start_date or not end_date:
            print("[ERROR] Revisa tu configs_ensemble.yaml.")
            return

    # 3. Descargar datos
    print(f"[INFO] Descargando datos desde Yahoo Finance...")
    df_raw = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df_raw = df_raw.reset_index()
    df_raw.columns = [c.lower() for c in df_raw.columns] 

    if 'date' in df_raw.columns:
        df_raw['date'] = pd.to_datetime(df_raw['date']).dt.tz_localize(None)
        df_raw = df_raw.set_index('date')
    
    # Solo pasamos números al modelo
    df_input = df_raw.select_dtypes(include=['number']).copy()

    # 4. Inicializar Ensemble y Generar Señal
    print("[INFO] Ejecutando modelos Ensemble (XGBoost, LSTM, etc.)...")
    # En prepare_data.py (dentro de generate_super_csv)
    # 1. Calculamos el split del 80%
    split_idx = int(len(df_raw) * 0.8)

    # 2. Se lo pasamos al ensemble
    ensemble = EnsembleModel(cfg)
    df_results = ensemble.fit_predict(df_raw, split_idx=split_idx) # <--- PASAR EL ÍNDICE


    # --- ARREGLO PARA DUPLICADOS Y FECHAS (Aquí estaba el error) ---
    # 1. Resetear índice con cuidado
    df_results = df_results.reset_index()
    
    # 2. Eliminar columnas duplicadas que causan el ValueError
    df_results = df_results.loc[:, ~df_results.columns.duplicated()].copy()

    # 3. Identificar la columna de fecha y renombrarla a 'Date' para el merge
    possible_date_cols = ['date', 'index', 'datetime', 'Date']
    for col in possible_date_cols:
        if col in df_results.columns:
            df_results = df_results.rename(columns={col: 'Date'})
            break

    # 4. Convertir 'Date' a datetime puro y normalizado
    df_results['Date'] = pd.to_datetime(df_results['Date']).dt.tz_localize(None).dt.normalize()

    # 5. Identificar señal
    if 'clean_ensemble' in df_results.columns:
        df_results['signal_ensemble'] = df_results['clean_ensemble']
    
    # 6. Unir con Sentimiento
    processed_path = f"data/processed/{symbol}_sentiment_finnhub.csv"
    
    if os.path.exists(processed_path):
        print(f"[INFO] Integrando archivo de sentimiento: {processed_path}")
        df_sent = pd.read_csv(processed_path)
        
        # Limpiamos duplicados en sentimiento también
        df_sent = df_sent.loc[:, ~df_sent.columns.duplicated()].copy()
        
        # Normalizamos fechas del sentimiento para el merge
        df_sent['Date'] = pd.to_datetime(df_sent['Date']).dt.tz_localize(None).dt.normalize()
        
        # Merge seguro
        df_final = pd.merge(df_results, df_sent[['Date', 'sentiment']], on='Date', how='left')
    else:
        print("[WARN] No se encontró archivo de sentimiento. Usando 0.0")
        df_final = df_results.copy()
        df_final['sentiment'] = 0.0

    # 7. Guardar el archivo final
    output_name = f"data/processed/{symbol}_hybrid_ready.csv"
    os.makedirs("data/processed", exist_ok=True)
    
    # Estandarizar a minúsculas
    df_final.columns = [c.lower() for c in df_final.columns]
    
    # Asegurar que todas las columnas necesarias existen (rellenar con 0 si faltan)
    cols_rl = ['date', 'open', 'high', 'low', 'close', 'volume', 'sentiment', 'signal_ensemble']
    for c in cols_rl:
        if c not in df_final.columns:
            df_final[c] = 0.0
    
    # Guardado limpio
    df_final[cols_rl].to_csv(output_name, index=False)
    print(f"\n[EXITO] Archivo listo para RL: {output_name}")
if __name__ == "__main__":
    generate_super_csv()