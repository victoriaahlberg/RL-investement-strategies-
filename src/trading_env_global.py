import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnvGlobal(gym.Env):
    """
    Unified trading environment (PPO / A2C / DQN compatible)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        use_sentiment=False,
        use_ensemble=False,
        initial_balance=10_000.0,
        window_size=10,
        commission=0.0015,
        lambda_efficiency=0.0
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)

        # =========================
        # AUTO FEATURE DETECTION
        # =========================
        base_cols = ["open", "high", "low", "close", "volume"]

        extra_cols = [
            "sentiment",
            "signal_ensemble",
            "prob_up",
            "prob_max_drawdown",
            "signal_entropy",
            "macd",
            "rsi",
            "ddi",
            "rolling_vol"
        ]

        self.feature_cols = base_cols.copy()

        if use_sentiment and "sentiment" in self.df.columns:
            self.feature_cols.append("sentiment")

        if use_ensemble and "signal_ensemble" in self.df.columns:
            self.feature_cols.append("signal_ensemble")

        for c in extra_cols:
            if c in self.df.columns and c not in self.feature_cols:
                self.feature_cols.append(c)

        # =========================
        # NUMPY ARRAYS (FAST)
        # =========================
        self.X = self.df[self.feature_cols].values.astype(np.float32)
        self.close = self.df["close"].values.astype(np.float32)

        self.n_steps, self.n_features = self.X.shape

        # =========================
        # PARAMS
        # =========================
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = commission
        self.lambda_efficiency = lambda_efficiency

        # =========================
        # SPACE
        # =========================
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * self.n_features + 3,),
            dtype=np.float32
        )

        self.reset()

    # =========================
    # RESET
    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares = 0.0
        self.step_idx = self.window_size
        self.net_worth = self.initial_balance
        self.trades = 0

        return self._get_obs(), {}

    # =========================
    # OBS
    # =========================
    def _get_obs(self):

        start = max(0, self.step_idx - self.window_size)
        end = self.step_idx

        window = self.X[start:end].flatten()

        # 🔴 si no hay datos suficientes, rellena
        if len(window) == 0:
            window = np.zeros(self.window_size * self.n_features, dtype=np.float32)

        price = self.close[self.step_idx - 1]

        obs = np.concatenate([
            window,
            np.array([self.balance, self.shares, self.balance + self.shares * price], dtype=np.float32)
        ]).astype(np.float32)

        return obs

    # =========================
    # STEP
    # =========================
    def step(self, action):
        price = self.close[self.step_idx]
        prev_price = self.close[self.step_idx - 1]

        prev_net = self.balance + self.shares * prev_price

        # ================= BUY =================
        if action == 1:
            cash = self.balance * 0.95
            shares = cash / price
            cost = shares * price
            fee = cost * self.commission

            self.balance -= (cost + fee)
            self.shares += shares
            self.trades += 1

        # ================= SELL =================
        elif action == 2:
            revenue = self.shares * price
            fee = revenue * self.commission

            self.balance += (revenue - fee)
            self.shares = 0.0
            self.trades += 1

        # avanzar tiempo
        self.step_idx += 1

        # ⚠️ IMPORTANTE: usar precio coherente tras step
        current_price = self.close[self.step_idx - 1]

        new_net = self.balance + self.shares * current_price
        self.net_worth = new_net

        # ================= REWARD =================
        log_return = np.log((new_net + 1e-8) / (prev_net + 1e-8))

        # ✔ mejor: penalización por trade incremental, no acumulada
        reward = log_return - self.lambda_efficiency * (1 if action != 0 else 0)

        done = self.step_idx >= self.n_steps - 1

        return self._get_obs(), float(reward), done, False, {
            "net_worth": self.net_worth,
            "trades": self.trades
        }