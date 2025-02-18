import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gym import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.model_selection import TimeSeriesSplit
import logging
import optuna
from stable_baselines3.common.callbacks import EvalCallback
import torch

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class ForexEnv(gym.Env):
    def __init__(self, data, max_steps=2500):
        super(ForexEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.max_steps = max_steps
        self.start_step = 0
        self.reward_history = []
        self.early_stop_patience = 200
        self.early_stop_threshold = -2

        num_features = self.data.shape[1]
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(15 * num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # [Buy, Sell, Hold]

    def reset(self):
        upper_bound = len(self.data) - self.max_steps
        self.start_step = np.random.randint(15, upper_bound) if upper_bound > 15 else 15
        self.current_step = self.start_step
        self.reward_history = []
        return self._get_observation()

    def _get_observation(self):
        start = max(0, self.current_step - 14)
        obs = self.data.iloc[start:self.current_step+1].values
        obs = np.pad(obs, ((15 - obs.shape[0], 0), (0, 0)), mode='constant')
        return obs.flatten()

    def step(self, action):
        if self.current_step >= len(self.data) - 16:
            return np.zeros(self.observation_space.shape), 0, True, {}

        current_close = self.data.iloc[self.current_step]['eurusd_close']
        future_prices = self.data.iloc[self.current_step+1 : self.current_step+16]['eurusd_close'].values
        future_mean_price = future_prices.mean()
        price_diff = future_mean_price - current_close

        reward = 0
        if action == 0:  # Buy
            reward = price_diff * 10000
        elif action == 1:  # Sell
            reward = -price_diff * 10000
        elif action == 2:  # Hold
            reward = -0.1

        self.reward_history.append(reward)
        if len(self.reward_history) > self.early_stop_patience:
            self.reward_history.pop(0)

        avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        done = (len(self.reward_history) == self.early_stop_patience and avg_reward < self.early_stop_threshold) or \
            (self.current_step - self.start_step >= self.max_steps) or \
            (self.current_step >= len(self.data) - 15)

        self.current_step += 1
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        return next_state, reward, done, {}

class ForexTradingModel:
    def __init__(self, data_, n_splits=5):
        self.data = data_
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.model = None

    def train_and_evaluate_single(self, train_data, val_data, params):
        train_env = ForexEnv(train_data)
        val_env = ForexEnv(val_data)
        activation_map = {
        'tanh': torch.nn.Tanh,
        'relu': torch.nn.ReLU
    }
        policy_kwargs = dict(activation_fn=activation_map[params['activation_fn']], 
                         net_arch=[params['n_layers'] * [params['n_neurons']]])
        
        self.model = PPO("MlpPolicy", train_env, verbose=0,
                         policy_kwargs=policy_kwargs,
                         learning_rate=params['learning_rate'],
                         n_steps=params['n_steps'],
                         batch_size=params['batch_size'],
                         n_epochs=params['n_epochs'],
                         gamma=params['gamma'],
                         clip_range=params['clip_range'])
        
        eval_callback = EvalCallback(val_env, eval_freq=10000, n_eval_episodes=5, 
                                     best_model_save_path='./best_model/', 
                                     log_path='./logs/', 
                                     deterministic=True)
        
        self.model.learn(total_timesteps=10000, callback=eval_callback)
        
        # Since we're using an EvalCallback, we'll get the best performance from the saved best model
        best_model = PPO.load('./best_model/best_model')
        mean_reward, std_reward = evaluate_policy(best_model, val_env, n_eval_episodes=5)
        performance = self.evaluate_trading_performance(best_model, val_env)
        
        return mean_reward

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

            if action in [0, 1]:  # Assuming 0 is buy, 1 is sell
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

    def objective(self, trial):
        n_steps = trial.suggest_int('n_steps', 128, 2048)
        params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_steps': n_steps,
        'batch_size': trial.suggest_int('batch_size', 32, min(512, n_steps)),
        'n_epochs': trial.suggest_int('n_epochs', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'n_neurons': trial.suggest_int('n_neurons', 32, 256),
        'n_layers': trial.suggest_int('n_layers', 1, 5),
        'activation_fn': trial.suggest_categorical('activation_fn', ['tanh', 'relu'])
    }
        
        mean_rewards = []
        for train_index, val_index in self.tscv.split(self.data):
            train_data, val_data = self.data.iloc[train_index], self.data.iloc[val_index]
            mean_reward = self.train_and_evaluate_single(train_data, val_data, params)
            mean_rewards.append(mean_reward)
        
        return np.mean(mean_rewards)

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        best_params = study.best_params
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best value: {study.best_value}")

if __name__ == "__main__":
    data = pd.read_csv('coin_df5.csv')
    data= data[['cos_time','sin_time','eurusd_close','eurusd_log_return','Normalized_eurusd_eurjpy_Coin','Normalized_eurusd_eurgbp_Coin','Normalized_eurusd_audjpy_Coin','Normalized_eurusd_audusd_Coin',
     'Normalized_eurusd_gbpjpy_Coin','Normalized_eurusd_nzdjpy_Coin','Normalized_eurusd_usdcad_Coin','Normalized_eurusd_usdchf_Coin',
     'Normalized_eurusd_usdhkd_Coin','Normalized_eurusd_usdjpy_Coin','Normalized_eurjpy_eurgbp_Coin','Normalized_eurjpy_audjpy_Coin',
     'Normalized_eurjpy_audusd_Coin','Normalized_eurjpy_gbpjpy_Coin','Normalized_eurjpy_nzdjpy_Coin','Normalized_eurjpy_usdcad_Coin',
     'Normalized_eurjpy_usdchf_Coin','Normalized_eurjpy_usdhkd_Coin','Normalized_eurjpy_usdjpy_Coin','Normalized_eurgbp_audjpy_Coin',
     'Normalized_eurgbp_audusd_Coin','Normalized_eurgbp_gbpjpy_Coin','Normalized_eurgbp_nzdjpy_Coin','Normalized_eurgbp_usdcad_Coin',
     'Normalized_eurgbp_usdchf_Coin','Normalized_eurgbp_usdhkd_Coin','Normalized_eurgbp_usdjpy_Coin','Normalized_audjpy_audusd_Coin',
     'Normalized_audjpy_gbpjpy_Coin','Normalized_audjpy_nzdjpy_Coin','Normalized_audjpy_usdcad_Coin','Normalized_audjpy_usdchf_Coin',
     'Normalized_audjpy_usdhkd_Coin','Normalized_audjpy_usdjpy_Coin','Normalized_audusd_gbpjpy_Coin','Normalized_audusd_nzdjpy_Coin',
     'Normalized_audusd_usdcad_Coin','Normalized_audusd_usdchf_Coin','Normalized_audusd_usdhkd_Coin','Normalized_audusd_usdjpy_Coin',
     'Normalized_gbpjpy_nzdjpy_Coin','Normalized_gbpjpy_usdcad_Coin','Normalized_gbpjpy_usdchf_Coin','Normalized_gbpjpy_usdhkd_Coin',
     'Normalized_gbpjpy_usdjpy_Coin','Normalized_nzdjpy_usdcad_Coin','Normalized_nzdjpy_usdchf_Coin','Normalized_nzdjpy_usdhkd_Coin',
     'Normalized_nzdjpy_usdjpy_Coin','Normalized_usdcad_usdchf_Coin','Normalized_usdcad_usdhkd_Coin','Normalized_usdcad_usdjpy_Coin',
     'Normalized_usdchf_usdhkd_Coin','Normalized_usdchf_usdjpy_Coin','Normalized_usdhkd_usdjpy_Coin']]
    trader = ForexTradingModel(data)
    trader.optimize(n_trials=50)  # Adjust the number of trials based on your computational resources

    # Optionally, you can then train with the best parameters found:
    best_params = study.best_params