
import numpy as np
import pandas as pd
from scipy.stats import entropy


#métricas para el agente
#returns= (P_t-P_t-1)/P_t-1 : market returns que NO dependen del agente 
#No confundir con los returns del agente (sería el Net Worth)

#prices : pd.Series - Serie de precios de cierre.


def prob_up(prices: pd.Series, horizon: int = 21) -> pd.Series:
    returns = prices.pct_change().fillna(0)

    prob_series = []

    for i in range(len(returns)):
        if i < horizon:
            prob_series.append(0.5)
        else:
            #para cada día mira en la ventana de los últimos días para ver cuantos de ellos fueron positivos
            # a partir de este número calcula la probabilidad de subida
            window = returns.iloc[i-horizon:i] #coge los últimos horizon valordes antes del día i
            prob = (window > 0).mean() #la media de los valores positivos del vector creado
            prob_series.append(prob)

    return pd.Series(prob_series, index=prices.index)

    
def signal_entropy(prices:pd.Series, horizon: int=21)-> pd.Series:
    returns=prices.pct_change().fillna(0)

#crea un histograma para calcular la entropía, si es baja el entorno es más predecible, si es alta hay más ruido
    def entropy_window(x):
        #queremos transformar datos en distribución de probabilidad
        hist, _= np.histogram(x, bins=10, density=True)
        hist+= 1e-8
        return entropy(hist) #mide como de repartidas están esas probabilidades 
    return returns.rolling(horizon).apply(entropy_window, raw=True).fillna(0)

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
    #resultado: probabilidad de estar en zona de caída fuerte 
    return pd.Series(mdd_prob, index=prices.index).fillna(0.0)

#Moving average convergence divergence 
def macd(prices:pd.Series, horizon_slow: int=21, horizon_fast: int=12, signal_period=9):
    #ewm es la media movil exponencial
    ema_fast=prices.ewm(span=horizon_fast, adjust=False).mean() #adjust=False: panda calcula la media de forma recuriva
    ema_slow=prices.ewm(span=horizon_slow, adjust=False).mean()

    #MACD line
    macd_line=ema_fast - ema_slow #dirección y fuerza de la tendencia 
    #signal
    signal_line=macd_line.ewm(span=signal_period, adjust=False).mean() #confirma señales de compra venta

    # Histograma MACD (lo que se suele usar como indicador)
    macd_hist = macd_line - signal_line

    # Rellenamos primeros valores con 0
    macd_hist = macd_hist.fillna(0.0)

    return macd_hist #pd.series que representa el histograma

def relative_strength(prices:pd.Series, horizon: int=21)->pd.Series:
    returns = prices.pct_change().fillna(0)

    gains = returns.clip(lower=0) #cambia todos los valors menores que 0 por 0
    losses = -returns.clip(upper=0)
    #calculamos la media movil para cada 
    #es una serie de valores, no un valor único
    ema_gains=gains.ewm(span=horizon, adjust=False).mean()
    ema_loss=losses.ewm(span=horizon, adjust=False).mean()
    #calcuamos RS
    rel_strength= ema_gains/(ema_loss + 1e-8) #evitar división por 0
    rsi= 100 - (100/1+rel_strength)

    return rsi

def ddi(high: pd.Series, low: pd.Series, close: pd.Series, horizon: int=14)->pd.Series:
    #aquí el horizon son los días necesarios para suavizar el EMA
    
    #directional movements
    dm_plus=high.diff() #High_t-High_{t-1}
    dm_minus=low.shift(1)-low  #Low_{t-1}-Low_t
 
    # Solo uno mantiene valor positivo
    dm_plus[dm_plus < 0] = 0
    dm_minus[dm_minus < 0] = 0
    mask = dm_plus > dm_minus
    dm_plus[~mask] = 0
    dm_minus[mask] = 0

    #calculating true range; máximo día a día
    tr= pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()],axis=1).max(axis=1)
    
    #media móvil
    di_plus = dm_plus.ewm(span=horizon, adjust=False).mean() / tr.ewm(span=horizon, adjust=False).mean()
    di_minus = dm_minus.ewm(span=horizon, adjust=False).mean() / tr.ewm(span=horizon, adjust=False).mean()
    
    #calculamos indicador DDI 
    ddi_series = (di_plus - di_minus) / (di_plus + di_minus)
    
    return ddi_series.fillna(0)

def rolling_volatility(prices:pd.Series, horizon: int=21)->pd.Series:
    returns = prices.pct_change().fillna(0)

    #el rolling window significa que para cada día t, vamos a mirar los últimos window valores de retorno
    #std: calculamos la desviacion estandar
    vol=returns.rolling(horizon).std().fillna(0)

    return vol


