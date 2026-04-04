import numpy as np
import pandas as pd
from collections import Counter
import logging

logger = logging.getLogger(__name__)

#métricas de comparación final, después de la simulación

def calculate_sharpe_ratio(net_worth: pd.Series)->float: #mencionar que asumimos un risk-free rate constante
      #rentabilidad entre riesgo
      if len(net_worth)<2:
           logger.warning ("Net worth series too short to calculate Sharpe Ratio")
           return 0.0
      #diff te calucla los returns de cada día
      returns= np.diff(net_worth)/net_worth [:-1]
      mean_return= np.mean(returns)
      std_return=np.std(returns)
      if std_return==0:
        logger.warning("Zero volatility in returns, sharpe ratio undefined")
        return 0.0
      sharpe_ratio= mean_return/std_return *np.sqrt(252)
      return sharpe_ratio

def calculate_max_drawdown(net_worth: pd.Series)->float:
   
    cumulative_max= np.maximum.accumulate(net_worth)
    drawdowns= (cumulative_max-net_worth)/cumulative_max
    return drawdowns.max()

def volatility(net_worth: pd.Series) -> float:
    if len(net_worth) < 2:
        return 0.0
    # pct_change calcula (P_t - P_{t-1}) / P_{t-1}
    returns = net_worth.pct_change().dropna()  # % cambio entre días consecutivos
    vol = np.std(returns) * np.sqrt(252)       
    return vol

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



