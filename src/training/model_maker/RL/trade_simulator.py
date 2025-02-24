import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import torch
import random
import matplotlib.pyplot as plt

# Load the trained model
model_path = "ppo_forex_model_v2.zip"  # Update with your actual model path
model = PPO.load(model_path)

# Load and prepare test data (last 15%)
data = pd.read_csv('coin_df6.csv')  # Update with your actual data path
split = int(len(data) * 0.85)
test_data = data.iloc[split:].copy()

# Detrend the log return (same as training)
test_data['detrended_log_return'] = test_data['eurusd_log_return'] - test_data['eurusd_log_return'].rolling(window=50, min_periods=1).mean()

# Add necessary features using detrended log return
test_data['log_return_std'] = test_data['detrended_log_return'].rolling(window=15, min_periods=1).std()
test_data['log_return_mean'] = test_data['detrended_log_return'].rolling(window=15, min_periods=1).mean()
test_data['log_return_std_long'] = test_data['detrended_log_return'].rolling(window=50, min_periods=1).std()
test_data['bb_middle'] = test_data['detrended_log_return'].rolling(window=15, min_periods=1).mean()
test_data['bb_upper'] = test_data['bb_middle'] + 2 * test_data['log_return_std']
test_data['bb_lower'] = test_data['bb_middle'] - 2 * test_data['log_return_std']
exp1 = test_data['detrended_log_return'].ewm(span=12, adjust=False).mean()
exp2 = test_data['detrended_log_return'].ewm(span=26, adjust=False).mean()
test_data['macd'] = exp1 - exp2
test_data['macd_signal'] = test_data['macd'].ewm(span=9, adjust=False).mean()
test_data['macd_diff'] = test_data['macd'] - test_data['macd_signal']

# Clean NaN or inf values
test_data = test_data.replace([np.inf, -np.inf], np.nan).dropna()

# Select training columns (updated to include detrended_log_return)
test_data = test_data[['log_return_mean', 'log_return_std', 'log_return_std_long', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'macd_diff',
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

class TradingSimulator:
    def __init__(self, data, initial_balance=10000, risk_percentage=0.005, min_position_size=0.01, max_position_size=5.0, spread_pips=0, slippage_std_pips=0.0, penalty=0, sl_factor=3, tp_factor=9):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage  # 0.5%
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.spread_pips = spread_pips
        self.slippage_std_pips = slippage_std_pips
        self.penalty = penalty
        self.sl_factor = sl_factor  # SL multiplier
        self.tp_factor = tp_factor  # TP multiplier for 3:1 ratio
        self.balance = initial_balance
        self.equity_curve = []
        self.trades = []
        self.current_step = 15
        self.active_trade = None
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None
        self.position_size = self.calculate_position_size()
        self.sl_pips = None
        self.tp_pips = None

    def calculate_position_size(self):
        target_risk_dollars = self.risk_percentage * self.balance
        position_size = target_risk_dollars / 10  # $10/pip per lot
        return max(self.min_position_size, min(self.max_position_size, position_size))

    def get_observation(self):
        start = max(0, self.current_step - 14)
        obs = self.data.iloc[start:self.current_step+1].values
        if len(obs) < 15:
            obs = np.pad(obs, ((15 - len(obs), 0), (0, 0)), mode='constant', constant_values=0)
        obs = obs.flatten().astype(np.float32)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def step(self):
        if self.current_step >= len(self.data) or self.balance < 0.2 * self.initial_balance:  # Kill switch at 20%
            return False

        obs = self.get_observation()
        print(f"Step {self.current_step}, Obs shape: {obs.shape}, Any NaN: {np.any(np.isnan(obs))}")

        # Convert observation to tensor and move to device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)

        # Get action distribution and probabilities
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]  # Probabilities for [Buy, Sell, Hold]
            action, _ = model.predict(obs, deterministic=True)

        print(f"Action: {action}, Probabilities: Buy={probs[0]:.4f}, Sell={probs[1]:.4f}, Hold={probs[2]:.4f}")

        if self.active_trade is None:
            if action == 0:  # Buy
                self.enter_trade('buy')
            elif action == 1:  # Sell
                self.enter_trade('sell')
        elif self.active_trade == 'buy':
            self.check_exit('buy')
        elif self.active_trade == 'sell':
            self.check_exit('sell')

        self.current_step += 1
        self.equity_curve.append(self.balance)
        return True

    def enter_trade(self, direction):
        self.active_trade = direction
        self.position_size = self.calculate_position_size()
        current_price = self.data.iloc[self.current_step]['eurusd_close']
        slippage = np.random.normal(0, self.slippage_std_pips) / 10000
        self.entry_price = current_price + slippage if direction == 'buy' else current_price - slippage
        
        past_log_returns = self.data.iloc[max(0, self.current_step - 14):self.current_step+1]['detrended_log_return'].values
        past_std = np.std(past_log_returns)
        self.sl_pips = self.sl_factor * past_std * 10000  # e.g., 15 pips if past_std ≈ 0.0005
        self.tp_pips = self.tp_factor * past_std * 10000  # e.g., 45 pips for 3:1 ratio
        
        if direction == 'buy':
            self.tp_price = self.entry_price + (self.tp_pips / 10000) - (self.spread_pips / 10000)
            self.sl_price = self.entry_price - (self.sl_pips / 10000)
        else:
            self.tp_price = self.entry_price - (self.tp_pips / 10000) + (self.spread_pips / 10000)
            self.sl_price = self.entry_price + (self.sl_pips / 10000)

    def check_exit(self, direction):
        current_price = self.data.iloc[self.current_step]['eurusd_close']
        slippage = np.random.normal(0, self.slippage_std_pips) / 10000
        effective_price = current_price - slippage if direction == 'buy' else current_price + slippage
        
        if direction == 'buy':
            if effective_price >= self.tp_price:
                self.close_trade('tp', effective_price)
            elif effective_price <= self.sl_price:
                self.close_trade('sl', effective_price)
        else:
            if effective_price <= self.tp_price:
                self.close_trade('tp', effective_price)
            elif effective_price >= self.sl_price:
                self.close_trade('sl', effective_price)

    def close_trade(self, reason, exit_price):
        if self.active_trade == 'buy':
            pips_gained_raw = (exit_price - self.entry_price) * 10000 - self.penalty
            if reason == 'tp':
                pips_gained = min(pips_gained_raw, self.tp_pips)
            elif reason == 'sl':
                pips_gained = max(pips_gained_raw, -self.sl_pips)
        else:
            pips_gained_raw = (self.entry_price - exit_price) * 10000 - self.penalty
            if reason == 'tp':
                pips_gained = min(pips_gained_raw, self.tp_pips)
            elif reason == 'sl':
                pips_gained = max(pips_gained_raw, -self.sl_pips)
        
        profit = pips_gained * self.position_size * 10
        self.balance += profit
        print(f'Current balance: {self.balance}, Position size: {self.position_size}, Profit: {profit}')
        self.trades.append({
            'direction': self.active_trade,
            'exit_step': self.current_step,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pips_gained': pips_gained,
            'profit': profit,
            'reason': reason,
            'position_size': self.position_size
        })
        self.active_trade = None
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None

# Run the simulation
simulator = TradingSimulator(
    test_data,
    initial_balance=10000,
    risk_percentage=0.005,
    min_position_size=0.01,
    max_position_size=5.0,
    spread_pips=0,  # Kept at 0 for consistency with your latest run
    slippage_std_pips=0.0,
    penalty=0,
    sl_factor=3,
    tp_factor=9  # 3:1 ratio
)
while simulator.step():
    pass

# Analyze results
print(f"Final Account Balance: {simulator.balance}")
print(f"Number of Trades: {len(simulator.trades)}")
wins = sum(1 for trade in simulator.trades if trade['profit'] > 0)
win_rate = wins / len(simulator.trades) if len(simulator.trades) > 0 else 0
print(f"Win Rate: {win_rate:.2%}")
equity_curve = np.array(simulator.equity_curve)
peak = np.maximum.accumulate(equity_curve)
drawdown = peak - equity_curve
max_drawdown = np.max(drawdown)
print(f"Max Drawdown: {max_drawdown}")

# Additional diagnostics
num_tp = sum(1 for trade in simulator.trades if trade['reason'] == 'tp')
num_sl = sum(1 for trade in simulator.trades if trade['reason'] == 'sl')
avg_tp_pips = np.mean([trade['pips_gained'] for trade in simulator.trades if trade['reason'] == 'tp']) if num_tp > 0 else 0
avg_sl_pips = np.mean([trade['pips_gained'] for trade in simulator.trades if trade['reason'] == 'sl']) if num_sl > 0 else 0
total_pips = sum(trade['pips_gained'] for trade in simulator.trades)

print(f"TP Hits: {num_tp}, SL Hits: {num_sl}")
print(f"Average TP Pips: {avg_tp_pips:.2f}, Average SL Pips: {avg_sl_pips:.2f}")
print(f"Total Pips Gained: {total_pips:.2f}")

# Plot equity curve
plt.plot(simulator.equity_curve)
plt.title('Equity Curve')
plt.xlabel('Steps')
plt.ylabel('Balance')
plt.show()