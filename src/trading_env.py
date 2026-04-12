"""
Simple Gymnasium trading environment for educational purposes (TFG).

Features
--------
- Discrete actions: 0 = hold, 1 = buy all cash, 2 = sell all shares
- Optional sentiment in observations
- Initial balance configurable
- No commissions (kept simple for learning)
- Safe indexing (handles episode end correctly)
- Compatible with Stable-Baselines3 PPO
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Dict, Any



class TradingEnv(gym.Env):
    """
    Custom trading environment for RL agents.

    The agent starts with cash and can buy/sell shares of one stock.
    Observation: price data (open, high, low, close, volume) + optional sentiment.
    Actions: hold, buy all, sell all.
    Reward: profit when selling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        use_sentiment: bool = True,
        initial_balance: float = 10_000.0,
        return_horizon: int = 21, #new time horizon added to improve hold actions
        window_size: int= 10
        ) -> None:
        """
        Initialize the environment.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: ['open', 'high', 'low', 'close', 'volume']
            Optional: 'sentiment'
            Index will be reset to 0..N for safe iloc access.
        use_sentiment : bool
            Include sentiment in observations.
        initial_balance : float
            Starting cash (e.g., 10,000 €).
        """
        super().__init__()

        # Clean data and reset index for safe iloc access
        self.df = df.dropna().reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty after dropping NaN")

        self.use_sentiment = use_sentiment and "sentiment" in self.df.columns
        self.initial_balance = float(initial_balance)
        self.return_horizon = return_horizon

        self.window_size= window_size
        # State variables
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.current_step = 0

        
        #Número de features de mercado por día
        n_features_per_day = 5  # open, high, low, close, volume
        if self.use_sentiment:
            n_features_per_day += 1

        # Features extras que agregamos cada día
        extra_features = 7  # prob_up, prob_max_drawdown, signal_entropy, MACD, RSI, DDI, vol

        # Observación total
        #información que viene del mercado 
        window_features = self.window_size * (n_features_per_day+extra_features)
        agent_features= 3
        total_features= window_features + agent_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        for col in ["prob_up", "prob_max_drawdown", "signal_entropy"]:
            if col not in self.df.columns:
                self.df[col] = 0.0
        


    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to initial state (day 0, full cash, no shares)."""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0.0
        return self._get_observation(), {}
    
    def _get_net_worth(self, price:float)->float:
            return self.balance + self.shares_held*price

    def _get_observation(self) -> np.ndarray:
        #construye el vector de estado que recibe el estado
        end = self.current_step + 1
        start = max(0, end - self.window_size)
        window = self.df.iloc[start:end]
        n_rows = len(window)

        obs_list = []

        # Precio y volumen (5 columnas)
        price_vol = window[["open","high","low","close","volume"]].values.flatten()
        # si no hay suficientes datos 
        if n_rows < self.window_size:
            price_vol = np.pad(price_vol, (0, (self.window_size - n_rows) * 5), 'constant')
        obs_list.extend(price_vol)

        # Sentiment (si aplica)
        if self.use_sentiment:
            sentiment = window["sentiment"].values.flatten()
            if n_rows < self.window_size:
                sentiment = np.pad(sentiment, (0, self.window_size - n_rows), 'constant')
            obs_list.extend(sentiment)

        # Features para el agente
        for col in ["prob_up", "prob_max_drawdown", "signal_entropy", "macd", "rsi", "ddi", "rolling_vol"]:
            if col not in self.df.columns:
                self.df[col] = 0.0  # inicializa si no existe
            feat = window[col].values.flatten()
            if n_rows < self.window_size:
                feat = np.pad(feat, (0, self.window_size - n_rows), 'constant')
            obs_list.extend(feat)
            
        # Estado del agente
        obs_list.append(self.balance)
        obs_list.append(self.shares_held)

        current_price = window.iloc[-1]["close"]
        net_worth = self._get_net_worth(current_price)
        obs_list.append(net_worth)
        
        obs_array = np.array(obs_list, dtype=np.float32)
        #limpiar valores problemáticos 
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)

        #  Verifica tamaño
        if obs_array.shape[0] != self.observation_space.shape[0]:
            raise ValueError(f"OBS SIZE MISMATCH: got {obs_array.shape[0]}, expected {self.observation_space.shape[0]}")

        return obs_array
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        #ejecuta una acción del agente 

        #vaso fin del dataset
        if self.current_step >= len(self.df):
            terminated = True
            truncated = False
            reward = 0.0
            return self._get_observation(), reward, terminated, truncated, {}

        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        current_price = float(self.df.iloc[self.current_step]["close"])
        net_worth_before = self._get_net_worth(current_price)
        reward = 0.0

        # Ejecutar acción
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price
            self.shares_held += shares_to_buy
            self.balance -= cost
        elif action == 2:  # Sell
            revenue = self.shares_held * current_price
            self.balance += revenue
            self.shares_held = 0.0

        # Avanzar al siguiente día
        self.current_step += 1
        #para clcular  el reward
        next_idx = min(self.current_step, len(self.df) - 1)
        next_price = float(self.df.iloc[next_idx]["close"])

        net_worth_after = self._get_net_worth(next_price)
        self.net_worth = net_worth_after
        reward = np.log(net_worth_after / net_worth_before)*100

        terminated = self.current_step >= len(self.df)
        truncated = False

        return self._get_observation(), float(reward), terminated, truncated, {}

 

    def render(self, mode: str = "human") -> None: #modo human es estándar de gym
        #imprime el estado actual del portofolio
        """Print current portfolio status."""
        if mode != "human":
            return
        #usa el precio del día anterior, evitando errores si current_Step==0
        price = self.df.iloc[self.current_step - 1]["close"] if self.current_step > 0 else self.df.iloc[0]["close"]
        net_worth = self.balance + self.shares_held * price
        print(
            f"Day {self.current_step:3d} | Price: {price:8.2f} | "
            f"Shares: {self.shares_held:6.0f} | Cash: {self.balance:10.2f} | "
            f"Net Worth: {net_worth:10.2f}"
        )



    def close(self) -> None:
        """Cleanup (no-op)."""
        pass