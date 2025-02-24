import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gym import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
import logging
import torch

# Configure logging
logging.basicConfig(filename='/mnt/training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class EntropyDecayCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EntropyDecayCallback, self).__init__(verbose)
        self.initial_ent_coef = 0.5
        self.final_ent_coef = 0.3  # Increased for more exploration
        self.total_timesteps = 5000000  # Extended training

    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        current_ent_coef = self.initial_ent_coef + (self.final_ent_coef - self.initial_ent_coef) * progress
        self.model.ent_coef = torch.tensor(current_ent_coef, device=self.model.device)
        return True

class ForexEnv(gym.Env):
    def __init__(self, data, max_steps=2500):
        super(ForexEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.max_steps = max_steps
        self.start_step = 0
        self.reward_history = []
        self.trades = 0
        self.early_stop_patience = 500
        self.early_stop_threshold = -0.5
        num_features = self.data.shape[1]
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(15 * num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # [Buy, Sell, Hold]

    def reset(self):
        upper_bound = len(self.data) - self.max_steps
        self.start_step = np.random.randint(15, upper_bound) if upper_bound > 15 else 15
        self.current_step = self.start_step
        self.reward_history = []
        self.trades = 0
        return self._get_observation()

    def _get_observation(self):
        start = max(0, self.current_step - 14)
        obs = self.data.iloc[start:self.current_step+1].values
        if len(obs) < 15:
            obs = np.pad(obs, ((15 - obs.shape[0], 0), (0, 0)), mode='constant')
        obs = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-8)  # Normalize
        obs = obs.flatten().astype(np.float32)
        return obs

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return np.zeros(self.observation_space.shape), 0, True, {}
        
        past_log_returns = self.data.iloc[max(0, self.current_step - 14):self.current_step+1]['detrended_log_return'].values
        past_std = np.std(past_log_returns) if len(past_log_returns) > 0 else 1e-8
        upper_barrier = 0.09 * past_std
        lower_barrier = -0.09 * past_std
        
        future_log_return = self.data.iloc[self.current_step + 1]['detrended_log_return']
        
        if action == 0:  # Buy
            reward = 3 if future_log_return > upper_barrier else -1
            self.trades += 1
        elif action == 1:  # Sell
            reward = 3 if future_log_return < lower_barrier else -1
            self.trades += 1
        elif action == 2:  # Hold
            reward = 0 if abs(future_log_return) < 0.03 * past_std else -0.01
        
        print(f"Step {self.current_step}, Action: {action}, Reward: {reward}")  # Debug action distribution
        
        self.reward_history.append(reward)
        if len(self.reward_history) > self.early_stop_patience:
            self.reward_history.pop(0)
        
        self.current_step += 1
        
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        done = (len(self.reward_history) == self.early_stop_patience and avg_reward < self.early_stop_threshold) or \
               (self.current_step - self.start_step >= self.max_steps) or \
               (self.current_step >= len(self.data) - 1)
        
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return next_state, reward, done, {}

class ForexTradingModel:
    def __init__(self, data_):
        self.data = data_
        self.model = None

    def train_and_evaluate(self, train_data, val_data):
        train_env = ForexEnv(train_data)
        val_env = ForexEnv(val_data)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        policy_kwargs = dict(net_arch=[512, 512, 256])
        self.model = PPO("MlpPolicy", train_env, verbose=1,
                         learning_rate=0.003,  # Increased for better convergence
                         n_steps=2048,  # Doubled for more updates
                         batch_size=256,
                         n_epochs=10,
                         gamma=0.94,
                         clip_range=0.3,
                         ent_coef=0.5,
                         device=device,
                         policy_kwargs=policy_kwargs)
        
        entropy_callback = EntropyDecayCallback()
        eval_callback = EvalCallback(val_env, eval_freq=10000, n_eval_episodes=5, best_model_save_path='/mnt/checkpoints/best_model', verbose=1)
        checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='/mnt/checkpoints/', name_prefix='ppo_forex', verbose=1)
        
        self.model.learn(total_timesteps=5000000, callback=[entropy_callback, eval_callback, checkpoint_callback])
        
        model_save_path = "/mnt/ppo_forex_model_v3.zip"
        self.model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
        
        mean_reward, std_reward = evaluate_policy(self.model, val_env, n_eval_episodes=5)
        performance = self.evaluate_trading_performance(self.model, val_env)
        
        return mean_reward, performance

    def evaluate_trading_performance(self, model, env):
        obs = env.reset()
        total_reward = 0
        trades = 0
        profit = 0
        loss = 0
        max_drawdown = 0
        equity_curve = []

        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            equity_curve.append(total_reward)

            if action in [0, 1]:
                trades += 1
                if reward > 0:
                    profit += 1
                else:
                    loss += 1

            if done:
                break

        max_equity = max(equity_curve)
        drawdowns = [max_equity - x for x in equity_curve]
        max_drawdown = max(drawdowns)

        win_rate = profit / trades if trades > 0 else 0
        sharpe_ratio = total_reward / np.std(equity_curve) if len(equity_curve) > 1 else 0

        return {
            'total_reward': total_reward,
            'trades': trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

if __name__ == "__main__":
    data = pd.read_csv('coin_df6.csv')
    
    data['detrended_log_return'] = data['eurusd_log_return'] - data['eurusd_log_return'].rolling(window=50, min_periods=1).mean()
    
    data['log_return_std'] = data['detrended_log_return'].rolling(window=15, min_periods=1).std()
    data['log_return_mean'] = data['detrended_log_return'].rolling(window=15, min_periods=1).mean()
    data['log_return_std_long'] = data['detrended_log_return'].rolling(window=50, min_periods=1).std()
    data['bb_middle'] = data['detrended_log_return'].rolling(window=15, min_periods=1).mean()
    data['bb_upper'] = data['bb_middle'] + 2 * data['log_return_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['log_return_std']
    exp1 = data['detrended_log_return'].ewm(span=12, adjust=False).mean()
    exp2 = data['detrended_log_return'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - test_data['macd_signal']
    
    data.dropna(inplace=True)
    
    data = data[['log_return_mean', 'log_return_std', 'log_return_std_long', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'macd_diff',
                 'cos_time', 'sin_time', 'eurusd_close', 'detrended_log_return',
                 'Normalized_eurusd_eurjpy_Coin', 'Normalized_eurusd_eurgbp_Coin', 'Normalized_eurusd_audjpy_Coin',
                 'Normalized_eurusd_audusd_Coin', 'Normalized_eurusd_gbpjpy_Coin', 'Normalized_eurusd_nzdjpy_Coin',
                 'Normalized_eurusd_usdcad_Coin', 'Normalized_eurusd_usdchf_Coin', 'Normalized_eurusd_usdhkd_Coin',
                 'Normalized_eurusd_usdjpy_Coin', 'Normalized_eurjpy_eurgbp_Coin', 'Normalized_eurjpy_audjpy_Coin',
                 'Normalized_eurjpy_audusd_Coin', 'Normalized_eurjpy_gbpjpy_Coin', 'Normalized_eurjpy_nzdjpy_Coin',
                 'Normalized_eurjpy_usdcad_Coin', 'Normalized_eurjpy_usdchf_Coin', 'Normalized_eurjpy_usdhkd_Coin',
                 'Normalized_eurjpy_usdjpy_Coin', 'Normalized_eurgbp_audjpy_Coin', 'Normalized_eurgbp_audusd_Coin',
                 'Normalized_eurgbp_gbpjpy_Coin', 'Normalized_eurgbp_nzdjpy_Coin', 'Normalized_eurgbp_usdcad_Coin',
                 'Normalized_eurgbp_usdchf_Coin', 'Normalized_eurgbp_usdhkd_Coin', 'Normalized_eurgbp_usdjpy_Coin',
                 'Normalized_audjpy_audusd_Coin', 'Normalized_audjpy_gbpjpy_Coin', 'Normalized_audjpy_nzdjpy_Coin',
                 'Normalized_audjpy_usdcad_Coin', 'Normalized_audjpy_usdchf_Coin', 'Normalized_audjpy_usdhkd_Coin',
                 'Normalized_audjpy_usdjpy_Coin', 'Normalized_audusd_gbpjpy_Coin', 'Normalized_audusd_nzdjpy_Coin',
                 'Normalized_audusd_usdcad_Coin', 'Normalized_audusd_usdchf_Coin', 'Normalized_audusd_usdhkd_Coin',
                 'Normalized_audusd_usdjpy_Coin', 'Normalized_gbpjpy_nzdjpy_Coin', 'Normalized_gbpjpy_usdcad_Coin',
                 'Normalized_gbpjpy_usdchf_Coin', 'Normalized_gbpjpy_usdhkd_Coin', 'Normalized_gbpjpy_usdjpy_Coin',
                 'Normalized_nzdjpy_usdcad_Coin', 'Normalized_nzdjpy_usdchf_Coin', 'Normalized_nzdjpy_usdhkd_Coin',
                 'Normalized_nzdjpy_usdjpy_Coin', 'Normalized_usdcad_usdchf_Coin', 'Normalized_usdcad_usdhkd_Coin',
                 'Normalized_usdcad_usdjpy_Coin', 'Normalized_usdchf_usdhkd_Coin', 'Normalized_usdchf_usdjpy_Coin',
                 'Normalized_usdhkd_usdjpy_Coin', 'RSI', '%K', '%D', 'direction', 'tenkan_sen', 'kijun_sen',
                 'senkou_span_a', 'senkou_span_b', 'chikou_span']]
    
    trader = ForexTradingModel(data)
    train_split = int(len(data) * 0.7)
    val_split = int(len(data) * 0.85)
    train_data = data.iloc[:train_split]
    val_data = data.iloc[train_split:val_split]
    
    mean_reward, performance = trader.train_and_evaluate(train_data, val_data)
    
    logging.info(f"Mean Reward (Validation): {mean_reward}")
    logging.info(f"Performance Metrics (Validation): {performance}")