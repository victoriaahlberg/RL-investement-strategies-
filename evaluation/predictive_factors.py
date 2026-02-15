
import numpy as np
import pandas as pd
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

def cumulative_cvar(net_worth:pd.Series, alpha:float=0.05)->pd.Series:
    #alpha is the risk level
    returns = net_worth.pct_change().fillna(0)
    cvar_series=[]
    for i in range (1, len(returns)+1):
        hist_returns= returns.iloc[:i]
        #calculate value at risk
        var=np.percentile(hist_returns, alpha*100)
        #mean of returns worse than or equal to VaR
        cvar= hist_returns[hist_returns <= var].mean()
        cvar_series.append(cvar)
    return pd.Series(cvar_series, index=net_worth.index)
