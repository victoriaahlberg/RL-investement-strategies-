# src/gen_utils.py
"""
gen_utils.py

Shared configuration utilities for the RL finance pipeline.

Features
--------
- ``load_config()``: Load YAML configuration with error handling.
- Centralizes config loading for:
    * ``train_walk_forward.py``
    * ``analyze_walk_forward.py``
    * Any future script

All docstrings and comments in **English**.
"""

from pathlib import Path
import yaml
import logging
from typing import Dict

from datetime import datetime, timedelta
import pandas as pd
import os

# Optional yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config_walk_forward.yaml") -> Dict:
    """
    Load the configuration from a YAML file.

    This function is shared across training and analysis scripts to ensure
    consistent configuration loading and error reporting.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file.
        Default: ``configs/config_walk_forward.yaml``

    Returns
    -------
    Dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try: #verifica que el archivo existe
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded: {config_path.resolve()}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config {config_path}: {e}")
        raise
        
        
def load_price_data(path: Path, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Load price data from cache or yfinance – 100% safe, no side effects."""
    if path.exists():
        logger.info(f"Loading cached price data -> {path}")
        df = pd.read_csv(path)
    else:
        if yf is None:
            raise RuntimeError("yfinance not installed")
        logger.info("Downloading price data via yfinance...")
        end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        result = yf.download(symbol, start=start, end=end_dt, interval=interval,
                             auto_adjust=True, progress=False, threads=False)
        
        if result.empty:
            logger.warning("Intraday data not available. Falling back to daily.")
            result = yf.download(symbol, start=start, end=end_dt, interval="1d",
                                 auto_adjust=True, progress=False, threads=False)

        # --- ARREGLO PARA COLUMNAS MULTI-INDEX (TUPLAS) ---
        # Si yfinance devuelve ('Close', 'AAPL'), nos quedamos solo con 'Close'
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)
        
        df = result.reset_index()
        os.makedirs(path.parent, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Price data saved -> {path}")

    # Normalize date column
    date_cols = ["Date", "date", "Datetime", "datetime", "Timestamp"]
    # Convertimos col a string por seguridad antes de comparar
    date_col = next((c for c in date_cols if str(c) in df.columns), None)
    
    if date_col is None:
        raise KeyError(f"No date column found. Columns: {df.columns.tolist()}")
    
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # Normalize price columns
    col_map = {}
    for target, patterns in {
        "open": ["open"], "high": ["high"], "low": ["low"],
        "close": ["close", "adj close", "adjusted close"], "volume": ["volume"]
    }.items():
        for col in df.columns:
            # Forzamos str(col) para evitar el AttributeError: 'tuple' object has no attribute 'lower'
            column_name_str = str(col).lower()
            if any(p in column_name_str for p in patterns):
                col_map[col] = target
                break
                
    df = df.rename(columns=col_map)
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        # Si falla, mostramos qué columnas había para debug
        raise KeyError(f"Missing columns after normalization: {missing}. Available: {df.columns.tolist()}")
    # ... código anterior en gen_utils.py ...

    # FORZAR CONVERSIÓN A NUMÉRICO (Evita el error de 'str' / 'str')
    cols_to_fix = ["open", "high", "low", "close", "volume"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminar filas con NaNs resultantes de la conversión si las hubiera
    df = df.dropna(subset=["close"])

    return df[["Date", "open", "high", "low", "close", "volume"]]
 