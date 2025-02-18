import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gym import spaces
from stable_baselines3.common.evaluation import evaluate_policy

# Custom Forex Trading Environment
class ForexEnv(gym.Env):
    def __init__(self, data, max_steps=250000):
        super(ForexEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.max_steps = max_steps  # Limit episode length
        self.start_step = 0  # Track where an episode started
        self.reward_history = []
        self.early_stop_patience = 50  # Number of steps to check for early stopping
        self.early_stop_threshold = -5  # If avg reward is below this, stop early

        num_features = self.data.shape[1] # Excluding the label column
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(15 * num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # [Buy, Sell, Hold]

    def reset(self):
        self.start_step = np.random.randint(15, len(self.data) - self.max_steps)
        self.current_step = self.start_step
        self.reward_history = []
        
        state = self._get_observation()
        return state

    def _get_observation(self):
        """Returns the last 15 timesteps as state."""
        start = max(0, self.current_step - 14)  # Ensure we don't go below index 0
        obs = self.data.iloc[start:self.current_step+1].values
        obs = np.pad(obs, ((15 - obs.shape[0], 0), (0, 0)), mode='constant')  # Pad if less than 15 steps
        return obs.flatten()  # Convert to 1D

    
    def step(self, action):
        """Execute a step based on agent action."""
        if self.current_step >= len(self.data) - 16:
            return np.zeros(self.observation_space.shape), 0, True, {}

        current_close = self.data.iloc[self.current_step]['eurusd_close']
        future_prices = self.data.iloc[self.current_step+1 : self.current_step+16]['eurusd_close'].values
        future_mean_price = future_prices.mean()
        price_diff = future_mean_price - current_close

        # Reward logic
        reward = 0
        if action == 0:  # Buy
            reward = price_diff * 10000  # Convert to pips
        elif action == 1:  # Sell
            reward = -price_diff * 10000  # Profit from price drops

        # Reward small penalty for holding
        elif action == 2:  # Hold
            reward = -0.1  

        # Track rewards
        self.reward_history.append(reward)
        if len(self.reward_history) > self.early_stop_patience:
            self.reward_history.pop(0)

        # Check stopping conditions
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        done = (len(self.reward_history) == self.early_stop_patience and avg_reward < self.early_stop_threshold) or \
            (self.current_step - self.start_step >= self.max_steps) or \
            (self.current_step >= len(self.data) - 15)

        self.current_step += 1
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        return next_state, reward, done, {}


       

       

 #   '''
    


# Load your volume bars and create spread
if __name__ == "__main__":
    data = pd.read_csv("coin_df4.csv")
    '''
    print(data.columns)
    for i in data.columns:
        print(i)
    '''
    '''
    data= data[['eurusd_normalized_close','eurusd_close','eurgbp_normalized_close','Normalized_eurusd_eurjpy_Coin','Normalized_eurusd_eurgbp_Coin','Normalized_eurusd_audjpy_Coin','Normalized_eurusd_audusd_Coin',
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
    '''
    
   # data = data[['eurusd_close','Normalized_eurusd_eurjpy_Coin','Normalized_eurusd_gbpjpy_Coin' ]]
    #data['Spread'] = data['PairA_Close'] - 1.5 * data['PairB_Close']  # Example
    data= data[['eurusd_close','eurusd_log_return','Normalized_eurusd_eurjpy_Coin','Normalized_eurusd_eurgbp_Coin','Normalized_eurusd_audjpy_Coin','Normalized_eurusd_audusd_Coin',
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

    # Create environment
    env = ForexEnv(data)

    # Train RL agent
    model = PPO("MlpPolicy", env, verbose=1)
    patience = 3  # Stop if reward doesn't improve after N evaluations
    best_reward = float('-inf')
    stagnant_epochs = 0
    '''
    for i in range(10):  # Adjust based on total timesteps you want
        model.learn(total_timesteps=100000)
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
        print(f"Iteration {i+1}: Mean Reward: {mean_reward}, Std Reward: {std_reward}")
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            stagnant_epochs = 0  # Reset counter if reward improves
            model.save("forex_model.h5")
        else:
            stagnant_epochs += 1  # Track how long it's been stagnant
        
        if stagnant_epochs >= patience:
            print(f"Early stopping triggered after {i+1} iterations. Best mean reward: {best_reward}")
            break
    '''
    model.learn(total_timesteps=400000)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)

    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    model.save("forex_model.h5")


    # Evaluate
    total_win = 0 
    total_reward = 0
    total_loss = 0
    datalen = 0
    obs = env.reset()
    for _ in range(0,len(data)):
        action, _ = model.predict(obs)
        print(f'action: {action}')
        obs, reward, done, _ = env.step(action)
        print(f'obs: {obs}, reward: {reward}, done: {done}, step:{_}')
        if reward > 0:
            total_win += 1
        elif reward < -1.0:
            total_loss += 1
        else:
            total_win += 0
        total_reward += reward
        print('total reward:',total_reward)
        print('total wins:', total_win)
        datalen += 1
        if datalen > 0 and total_win > 0 and total_reward >0 and total_loss > 0:
            win_ratio = total_win / datalen
            loss_ratio = total_loss /datalen
            print('win ratio:', win_ratio)
            print('loss ratio:',loss_ratio)
            
            pp_trade = total_reward /datalen
            print('piptrade:', pp_trade)
        
            print('pipwin:',total_reward/total_win )
        
        if done:
            break
       # print('tot')
    #'''
        
    ''''
    def step(self, action):
        # Determine the price at the current step and a future step (e.g., 5 steps ahead)
        current_eurusd = self.data.iloc[self.current_step]['eurusd_close']
        future_eurusd = self.data.iloc[self.current_step + 10]['eurusd_close']
        
        # Use the difference over a window as part of your reward
        reward = 0
        #cumulative_reward = 0
       # window = 10
        if action == 0:  # Buy EURUSD
            reward = (future_eurusd - current_eurusd) * 10000
            
            for i in range(1, window + 1):
                future_price = self.data.iloc[self.current_step + i]['eurusd_close']
                cumulative_reward += (future_price - current_eurusd) * 10000  # adjust sign per action if needed
            
        
        elif action == 1:  # Sell EURUSD
            reward = (current_eurusd - future_eurusd) * 10000
            
            for i in range(1, window + 1):
                future_price = self.data.iloc[self.current_step + i]['eurusd_close']
                cumulative_reward += ( current_eurusd - future_price) * 10000  # adjust sign per action if needed
            if cumulative_reward > 0:
                reward += 1
            elif cumulative_reward <= 0:
                reward -= 1
            
        else: reward =0
        # ... similar for other actions

      
        # You might mix immediate reward and cumulative reward
       # reward =  cumulative_reward

        self.current_step += 1
        done = self.current_step >= len(self.data) - 10
        return self.data.iloc[self.current_step].values, reward, done, {}
    '''
    '''
    def step(self, action):
        ####EURUSD
        current_eurusd = self.data.iloc[self.current_step]['eurusd_close']
        #next_price = self.data.iloc[self.current_step + 5]['eurusd_close']
        eurusd_price = self.data.iloc[self.current_step + 5]['eurusd_close']

       
        ####EURGBP
        current_eurgbp = self.data.iloc[self.current_step]['eurgbp_close']
        #next_price = self.data.iloc[self.current_step + 5]['eurusd_close']
        eurgbp_price = self.data.iloc[self.current_step + 5]['eurgbp_close']

        reward = 0
        #next_price = next_price.mean()
        #print('next price:', next_price)
        if action == 0:  # Buy EURUSD
            
            
            reward = eurusd_price - current_eurusd
            reward = reward *10000
            reward = reward
            self.current_step += 1
        elif action == 1:  # SELL EURUSD
            reward = current_eurusd - eurusd_price
            reward = reward * 10000
            reward= reward
            self.current_step += 1
        elif action ==2: #SELL EURGBP
            reward = current_eurgbp - eurgbp_price
            reward = reward * 10000
            reward= reward
            self.current_step += 1
        elif action ==3: #BUY EURGBP
            reward = eurgbp_price - current_eurgbp
            reward = reward * 10000
            reward= reward
            self.current_step += 1

        else: self.current_step += 1
        done = self.current_step >= len(self.data) - 5
        return self.data.iloc[self.current_step].values, reward, done, {}
    '''