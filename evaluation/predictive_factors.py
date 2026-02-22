
import numpy as np
import pandas as pd
from scipy.stats import entropy

#returns are the base for all other risk measures
def cumulative_returns(net_worth:pd.Series)-> pd.Series:
    #computes for eveyr day
    returns= net_worth.pct_change().fillna(0) #sets the first days return to 0
    return returns

def cumulative_volatility(net_worth:pd.Series)->pd.Series:
    returns= cumulative_returns(net_worth)
    #expaniding.std is the cumulative standard deviation
    cum_vol=returns.expanding().std()
    return cum_vol

def cumulative_drawdown(net_worth:pd.Series)->pd.Series:
    running_max= net_worth.cummax()
    drawdown=(net_worth- running_max)/running_max
    return drawdown

#cumualtive value at risk
def cumulative_cvar(net_worth:pd.Series, alpha:float=0.05)->pd.Series:
    #alpha is the risk level
    returns = net_worth.pct_change().fillna(0)
    cvar_series=[]
    for i in range (1, len(returns)+1):
        hist_returns= returns.iloc[:i] #todos los retornos hasta el día i
        #calculate value at risk
        var=np.percentile(hist_returns, alpha*100)
        #mean of returns worse than or equal to VaR
        cvar= hist_returns[hist_returns <= var].mean()
        cvar_series.append(cvar)
    return pd.Series(cvar_series, index=net_worth.index)

def prob_up(net_worth: pd.Series, horizon:int=21)-> pd.Series:
    #probabilidad de subida en los próximos 21 días (los que tengamos definidos)
    result=[]
    for i in range (len(net_worth)):
        if i + horizon >= len(net_worth):
            result.append(np.nan) #si no hay datos futuros poner Nan
            continue
        future_return= net_worth.iloc[i+horizon]/net_worth.iloc[i]-1
        result.append(1.0 if future_return > 0 else 0.0) #1 subida, 0 bajada
    return pd.Series(result, index=net_worth.index)
    
def signal_entropy(net_worth:pd.Series, horizon: int=21)-> pd.Series:
    returns=net_worth.pct_change(periods=horizon).fillna(0)
    hist,_=np.histogram(returns, bins=10, density=True)#histrograma normalizado
    hist += 1e-8 #evitar log 0
    return pd.Series(entropy(hist), index=net_worth.index)


def prob_max_drawdown(prices: pd.Series, horizon: int = 21, threshold: float = 0.1) -> pd.Series:
    mdd_prob = []

    for i in range(len(prices)):
        if i < horizon:
            # No hay suficientes datos para la ventana, usamos NaN temporal
            mdd_prob.append(np.nan)
        else:
            # Ventana de precios para los últimos 'horizon' días
            window_prices = prices.iloc[i - horizon + 1 : i + 1]
            # Max acumulado en la ventana
            running_max = window_prices.cummax()
            # Drawdowns relativos
            drawdowns = (window_prices - running_max) / running_max
            # Probabilidad de que haya un drawdown <= -threshold
            prob = (drawdowns <= -threshold).mean()
            mdd_prob.append(prob)

    # Convertimos a Series y rellenamos NaNs iniciales con 0 (opcional)
    return pd.Series(mdd_prob, index=prices.index).fillna(0.0)
