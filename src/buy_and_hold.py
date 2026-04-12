import pandas as pd
import numpy as np
from evaluation.evaluation_metrics import calculate_sharpe, calculate_max_drawdown, volatility, num_trades
import logging
import os

logger= logging.getLogger(__name__)

def buy_and_hold(df: pd.DataFrame, initial_balance:float=10000):
    #esta estrategia consiste en comprar todo el primer día y mantener hasata el final
    #df (pd.DataFrame) debe tener columna close y Date como índice
    #devueñve pd.DataFrame con columnas net_worth y action

    #creamos un dataframe vacío que tendrá eñ net_worth diario y la acción realizada
    simulation_df = pd.DataFrame(index=df.index, columns=['net_worth', 'action'], dtype=float)
    
    first_price= df.iloc[0]['close'] #precio de cierre del primer día disponible
    shares= initial_balance // first_price #cuantas acciones se pueden comprar con el capital 
    cash = initial_balance - shares*first_price #lo que no se ha podido invertir (lo de arriba usa floor division)

    simulation_df['action'] = 0          # primero ponemos todos en hold
    simulation_df.loc[df.index[0], 'action'] = 1  # compramos solo el primer día

    #iteración sobre todos los días
    for date, row in df.iterrows():
        price = row['close'] #precio del día actual
        net_worth_absolute = cash + shares * price 
        net_worth_pct = 100 * (net_worth_absolute / initial_balance - 1)
        simulation_df.loc [date, 'net_worth']=net_worth_absolute #guardamos el net_worth diario en simulation_df
    
    #cambios entre dias consecutivos
    returns = np.log(simulation_df['net_worth'] / simulation_df['net_worth'].shift(1)).fillna(0.0)

    sharpe= calculate_sharpe(simulation_df['net_worth'])
    mdd= calculate_max_drawdown(simulation_df['net_worth'])
    vol= volatility(returns)
    trades= num_trades (simulation_df['action'])

    #lo guardamos aquí para que sea más fácil comparar 
    metrics ={
        'sharpe': sharpe,
        'max_drawdown': mdd,
        'volatility': vol,
        'num_trades': trades
    }
    #reporta en consola los resultados 
    logger.info(f"Buy & Hold Metrics:Sharpe={sharpe:.2f}, MDD={mdd:.1%}, Vol={vol:.1%}, Trades={trades}")
    final_net_worth = simulation_df['net_worth'].iloc[-1]

    return simulation_df, metrics, simulation_df['action'], final_net_worth
 