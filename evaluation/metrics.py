import numpy as np
import pandas as pd
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def calculate_sharpe_ratio(net_worth: pd.Series)->float:
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



