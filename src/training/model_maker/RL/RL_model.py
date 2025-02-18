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
        # Calculate the upper bound for selecting a start index
        upper_bound = len(self.data) - self.max_steps
        if upper_bound <= 15:
            # If data is too short relative to max_steps, just start at 15
            self.start_step = 15
        else:
            self.start_step = np.random.randint(15, upper_bound)
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
    data = pd.read_csv("coin_df5.csv")
    data_val = data.iloc[250000:]
    data = data.iloc[:200000]
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

    validation_data = data.iloc[-50000:].reset_index(drop=True)
        
    training_data = data.iloc[:-10000].reset_index(drop=True)

    # Create environment
    print("Training data shape:", training_data.shape)
    print("Validation data shape:", validation_data.shape)
    
    # Create environments for training and validation
    train_env = ForexEnv(training_data)
    val_env = ForexEnv(validation_data)
    
    # Train RL agent on training environment
    model = PPO("MlpPolicy", train_env, verbose=1)
    patience = 3  # early stopping patience based on evaluation improvement
    best_reward = float('-inf')
    stagnant_epochs = 0
    
    for i in range(10):  # Adjust iterations/total_timesteps as needed
        model.learn(total_timesteps=100000)
        
        # Evaluate on the training environment
        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=4)
        print(f"Iteration {i+1}: Mean Reward (Train): {mean_reward}, Std Reward: {std_reward}")
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            stagnant_epochs = 0
            model.save("forex_model2.h5")
        else:
            stagnant_epochs += 1
        if stagnant_epochs >= patience:
            print(f"Early stopping triggered after {i+1} iterations. Best mean reward: {best_reward}")
            break
    
    # After training, evaluate the model on the validation environment.
    val_mean_reward, val_std_reward = evaluate_policy(model, val_env, n_eval_episodes=10)
    print(f"Validation Mean Reward: {val_mean_reward}, Validation Reward Std: {val_std_reward}")
    
    # Optionally, run a custom evaluation loop on the validation environment:
    obs = val_env.reset()
    total_rewards = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = val_env.step(action)
        total_rewards += reward
    print("Total reward on one validation episode:", total_rewards)
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