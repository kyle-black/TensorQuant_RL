import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gym import spaces
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gym import spaces

# Custom Forex Trading Environment
class ForexEnv(gym.Env):
    def __init__(self, data, max_steps=100000):
        super(ForexEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.max_steps = max_steps  # Limit episode length
        self.start_step = 0  # Track where an episode started

        # Define state and action space
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(data.shape[1]-1,))
        self.action_space = spaces.Discrete(3)  # [Buy, Sell, Hold]

    def reset(self):
        """Reset environment at the start of an episode."""
        self.start_step = np.random.randint(0, len(self.data) - self.max_steps)  # Start at a random index
        self.current_step = self.start_step
        state = self.data.iloc[self.current_step].drop(['eurusd_close']).values
        return state

    def step(self, action):
        """Execute a step based on agent action."""
        current_eurusd = self.data.iloc[self.current_step]['eurusd_close']

        # Define future prices window
        future_prices = self.data.iloc[self.current_step+1 : self.current_step+16]['eurusd_close']

        # Compute standard deviation-based barriers
        past_prices = self.data.iloc[max(0, self.current_step-15):self.current_step]['eurusd_close']
        pct_changes = past_prices.pct_change().dropna()
        pct_change_std = pct_changes.std()
        amount_chg = current_eurusd * (2*pct_change_std)
        upper_barrier = current_eurusd + amount_chg
        lower_barrier = current_eurusd - amount_chg

        # Check if price hits a barrier first
        hit_upper = hit_lower = False
        for price in future_prices:
            if price >= upper_barrier:
                hit_upper = True
                break
            elif price <= lower_barrier:
                hit_lower = True
                break

        # Define rewards
        reward = 0
        if action == 0 and hit_upper:  # Buy and hits upper barrier
            reward = 1
        elif action == 0 and hit_lower:  # Buy and hits lower barrier
            reward = -1
        
        elif action == 1 and hit_lower:  # Sell and hits lower barrier
            reward = 1
        elif action == 1 and hit_upper:  # Sell and hitupper barrier
            reward = -1
        
        else: reward=0

       

        self.current_step += 1
        done = (self.current_step - self.start_step >= self.max_steps) or (self.current_step >= len(self.data) - 15)
        next_state = self.data.iloc[self.current_step].drop(['eurusd_close']).values if not done else np.zeros_like(self.observation_space.shape)

        return next_state, reward, done, {}


 #   '''
    '''
    def step(self, action):
        current_eurusd = self.data.iloc[self.current_step]['eurusd_close']
        
        # Compute barriers using standard deviation over the past 15 bars
        past_prices = self.data.iloc[max(0, self.current_step - 15):self.current_step]['eurusd_close']
        print('past prices', past_prices)
        pct_changes = past_prices.pct_change(periods=15)
        print('pct_changes', pct_changes)
        pct_change_std = pct_changes.std()
        amount_chg = current_eurusd * pct_change_std
        upper_barrier = current_eurusd + amount_chg
        lower_barrier = current_eurusd - amount_chg

        # Find which barrier is hit first
        future_prices = self.data.iloc[self.current_step:self.current_step + 15]['eurusd_close'].values
        first_hit = None  # Track which barrier is hit first
        print(future_prices)
        print(upper_barrier)
        print(lower_barrier)
        for price in future_prices:
            if price >= upper_barrier:
                first_hit = 'upper'
                break
            elif price <= lower_barrier:
                first_hit = 'lower'
                break
       # print(first_hit)
        # Assign reward based on which barrier is hit first
        reward = 0
        if action == 0:  # Buy
            if first_hit == 'upper':
                reward = 10  # Profit when hitting the upper barrier first
            elif first_hit == 'lower':
                reward = -10  # Loss when hitting the lower barrier first
        elif action == 1:  # Sell
            if first_hit == 'lower':
                reward = 10  # Profit when hitting the lower barrier first
            elif first_hit == 'upper':
                reward = -10  # Loss when hitting the upper barrier first
        
        # Small penalty if no barrier is hit within 15 steps
       # if first_hit is None:
        #    reward = -2  

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 15
        next_state = self.data.iloc[self.current_step].drop(['eurusd_close']).values
        
        return next_state, reward, done, {}
    '''
    '''
    def step(self, action):
        # Compute immediate reward as before

        past_eurusd_prices = self.data.iloc[self.current_step:self.current_step - 15]['eurusd_close']
        pct_changes = past_eurusd_prices.pct_change(15)
        pct_change_std = pct_changes.std()
        current_eurusd = self.data.iloc[self.current_step]['eurusd_close']
        amount_chg =current_eurusd * pct_change_std
        upper_barrier = current_eurusd + amount_chg
        lower_barrier = current_eurusd - amount_chg

        self.data.iloc[self.current_step:self.current_step + 15]['eurusd_close']

      #  future_eurusd = self.data.iloc[self.current_step + 5]['eurusd_close']

      # current_eurgbp = self.data.iloc[self.current_step]['eurgbp_close']
       # future_eurgbp = self.data.iloc[self.current_step + 5]['eurgbp_close']

       # eurgbp_prices = self.data.iloc[self.current_step+5:self.current_step + 15]['eurgbp_close']
       # future_eurgbp = eurgbp_prices.mean() 
       
       
       # prices = self.data.iloc[self.current_step:self.current_step + 5]['eurusd_close']
       # max_drawdown = (prices.max() - prices.min()) * 10000
        reward = 0
        if action == 0:  # Buy EURUSD
            reward = (future_eurusd - current_eurusd) * 10000
        elif action == 1:  # Sell EURUSD
            reward = (current_eurusd - future_eurusd) * 10000
       # elif action == 2:  # Sell GPBUSD
        #    reward = (current_eurgbp - future_eurgbp) * 10000
       # elif action == 3:  # buy GPBUSD
        #    reward = ( future_eurgbp -current_eurgbp ) * 10000
        # Suppose we also track the max drawdown in the next 5 steps as a risk metric
       # prices = self.data.iloc[self.current_step:self.current_step + 5]['eurusd_close']
      #  max_drawdown = (prices.max() - prices.min()) * 10000

        # Combine profit and risk: reward = profit - penalty * (drawdown)
       # risk_penalty = 0.5  # adjust this weight
       # reward = reward - risk_penalty * max_drawdown

        self.current_step += 1
        done = self.current_step >= len(self.data) - 15
        next_state = self.data.iloc[self.current_step].drop(['eurusd_close']).values
        return next_state, reward, done, {}
    '''

    



# Load your volume bars and create spread
if __name__ == "__main__":
    data = pd.read_csv("coin_df3.csv")
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
    data= data[['eurusd_close','Normalized_eurusd_eurjpy_Coin','Normalized_eurusd_eurgbp_Coin','Normalized_eurusd_audjpy_Coin','Normalized_eurusd_audusd_Coin',
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
  #  model.learn(total_timesteps=500000)
    print('now running new model   steps 100000 ##########')
    for i in range(10):  # Adjust based on total timesteps you want
        model.learn(total_timesteps=100000)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
        print(f"Iteration {i+1}: Mean Reward: {mean_reward}, Std Reward: {std_reward}")
        model.save("forex_model.h5")
   # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    #print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

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
        elif reward < 0:
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