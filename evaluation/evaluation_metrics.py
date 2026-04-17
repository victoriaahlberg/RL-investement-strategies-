import numpy as np
import pandas as pd
from collections import Counter
import logging

logger = logging.getLogger(__name__)

#métricas de comparación final, después de la simulación

def calculate_sharpe(net_worth: pd.Series, freq: str = "1d") -> float:
    """
    Sharpe ratio robusto adaptado a la frecuencia.
    """

    # Limpieza
    series = pd.to_numeric(net_worth, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) < 2:
        return 0.0

    # Returns simples
    returns = series.pct_change().dropna()

    if returns.std() == 0:
        return 0.0

    # Factor correcto según frecuencia
    ann_map = {
        "1d": 252,
        "1h": 252 * 7,
        "1m": 252 * 7 * 60
    }

    ann_factor = ann_map.get(freq, 252)

    sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor)

    return float(sharpe)

def calculate_max_drawdown(net_worth: pd.Series)->float:
    # Aseguramos que no haya valores menores o iguales a cero para evitar errores
    if (net_worth <= 0).any():
        return 1.0 
    
    rolling_max = net_worth.cummax()
    drawdowns = (net_worth - rolling_max) / rolling_max
    return float(drawdowns.min()) # Devuelve un valor negativo (ej: -0.15 para 15%)

def volatility(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors='coerce').dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(252))

def num_trades (actions: pd.Series) -> int:
    return np.sum(actions!=0)

def total_returns (net_worth:pd.Series)-> float:
     
    if len(net_worth) < 2:
          return 0.0
     
    initial_value= net_worth.iloc[0]
    final_value=net_worth.iloc[-1]

    total_ret= (final_value-initial_value)/initial_value
    return total_ret

def win_rate(net_worth:pd.Series, actions:pd.Series)->float:
    trades=[]
    entry_value=None

    for i in range(len(actions)):
        action = actions.iloc[i]

        if action == 1 and entry_value is None:
            entry_value = net_worth.iloc[i] #valor del portfolio al entrar 

        elif action == 2 and entry_value is not None: #cerrar el trade, sell
            exit_value = net_worth.iloc[i]
            trades.append(exit_value - entry_value)
            entry_value = None

    if len(trades) == 0:
        return 0.0

    wins = sum(1 for t in trades if t > 0)
    return wins / len(trades)

def calmar_ratio(net_worth: pd.Series) -> float:
    
    #Calmar Ratio = CAGR / Max Drawdown
    
    # Duración en años (días / 252)
    years = len(net_worth) / 252
    if years <= 0:
        return 0.0

    # CAGR: ((Vf / Vi) ** (1/n)) - 1
    cagr = (net_worth.iloc[-1] / net_worth.iloc[0]) ** (1 / years) - 1

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(net_worth)
    drawdowns = (cumulative_max - net_worth) / cumulative_max
    max_dd = drawdowns.max()

    if max_dd == 0:
        return np.inf  # Si nunca hay drawdown → Calmar infinito
    return cagr / max_dd


def calculate_final_net_worth(self) -> float:
    if self.current_step == 0:
        price = self.df.iloc[0]["close"]
    else:
        price = self.df.iloc[self.current_step - 1]["close"]

    return self.balance + self.shares_held * price

def annualized_return(net_worth: pd.Series) -> float: #porcentajr anual más o menos 
    if len(net_worth) < 2:
        return 0.0
    
    # --- Cambio aquí: calculamos años por longitud de la serie ---
    num_days = len(net_worth)
    years = num_days / 252  # 252 días bursátiles aprox por año
    
    if years <= 0:
        return 0.0
    return (net_worth.iloc[-1] / net_worth.iloc[0]) ** (1 / years) - 1

def count_trades(position: pd.Series) -> int:
    pos = position.fillna(0)
    return int((pos.diff().abs() > 0).sum())