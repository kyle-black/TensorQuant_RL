import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
import features

class DataGather:
    def __init__(self, forex_pairs):
        self.forex_pairs = forex_pairs
        self.combo_df = pd.DataFrame()

    def grab_raw_data(self):
        """Loads raw forex data from CSV."""
        self.raw_data = pd.read_csv('data/all_forex_pull_15_min.csv')
        ###########  Turn time to COS and SIN data
        period = 86400  # 24 hours

        timestamps = self.raw_data['date']
        # Compute sine and cosine
        sin_time = np.sin(2 * np.pi * (timestamps % period) / period)
        cos_time = np.cos(2 * np.pi * (timestamps % period) / period)

        
        
        
        self.raw_data['sin_time'] = sin_time
        self.raw_data['cos_time'] = cos_time

        window_length =15
        num_std_dev = 2
        eurusd_close ='eurusd_close'
        self.raw_data['Middle_Band'] = self.raw_data[eurusd_close].rolling(window=window_length).mean()
        self.raw_data['Upper_Band'] = self.raw_data['Middle_Band'] + self.raw_data[eurusd_close].rolling(window=window_length).std() * num_std_dev
        self.raw_data['Lower_Band'] = self.raw_data['Middle_Band'] - self.raw_data[eurusd_close].rolling(window=window_length).std() * num_std_dev

        # Log Returns
        self.raw_data['Log_Returns'] = np.log(self.raw_data[eurusd_close]/ self.raw_data[eurusd_close].shift(window_length))

        ####################### MACD 
        exp1 = self.raw_data[eurusd_close].ewm(span=12, adjust=False).mean()
        exp2 = self.raw_data[eurusd_close].ewm(span=26, adjust=False).mean()
        self.raw_data['MACD'] = exp1 - exp2
        self.raw_data['Signal_Line_MACD'] = self.raw_data['MACD'].ewm(span=9, adjust=False).mean()

        ######################### RSI
        delta = self.raw_data[eurusd_close].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window_length).mean()
        avg_loss = loss.rolling(window=window_length).mean()
        rs = avg_gain / avg_loss
        self.raw_data['RSI'] = 100 - (100 / (1 + rs))

        low_min  = self.raw_data[eurusd_close].rolling(window=window_length).min()
        high_max = self.raw_data[eurusd_close].rolling(window=window_length).max()

        self.raw_data['%K'] = (self.raw_data[eurusd_close] - low_min) / (high_max - low_min) * 100
        self.raw_data['%D'] = self.raw_data['%K'].rolling(window=window_length).mean()

        self.raw_data['daily_return'] = self.raw_data[eurusd_close].diff()
        self.raw_data['direction'] = np.where(self.raw_data['daily_return'] > 0, 1, -1)
        self.raw_data.loc[self.raw_data['daily_return'] == 0, 'direction'] = 0

        period9_high = self.raw_data[eurusd_close].rolling(window=9).max()
        period9_low = self.raw_data[eurusd_close].rolling(window=9).min()
        self.raw_data['tenkan_sen'] = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = self.raw_data[eurusd_close].rolling(window=26).max()
        period26_low = self.raw_data[eurusd_close].rolling(window=26).min()
        self.raw_data['kijun_sen'] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        self.raw_data['senkou_span_a'] = ((self.raw_data['tenkan_sen'] + self.raw_data['kijun_sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = self.raw_data[eurusd_close].rolling(window= 52).max()
        period52_low = self.raw_data[eurusd_close].rolling(window=52).min()
        self.raw_data['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

        # Chikou Span (Lagging Span): eurusd_close shifted back 26 periods
        self.raw_data['chikou_span'] = self.raw_data[eurusd_close].shift(26)


        
        
        return self.raw_data
    
    def generate_combos(self):
        """Generates all forex pair combinations."""
        return list(combinations(self.forex_pairs, 2))
    
    def combos_to_df(self):
        """Computes log returns for forex pairs and stores pair combinations."""
        self.combos = self.generate_combos()
        
        # Compute log returns
        for pair in self.forex_pairs:
            self.raw_data[f'{pair}_log_return'] = np.log(self.raw_data[f'{pair}_close']) - np.log(self.raw_data[f'{pair}_close'].shift(1))
        
        return self.raw_data

    def cointegration_spread(self):
        """Computes cointegration spread using PCA and retains log returns."""
        coint_data = []  

        sin_time =self.raw_data['sin_time']
        cos_time = self.raw_data['cos_time'] 
        RSI = self.raw_data['RSI']
        K_ = self.raw_data['%K']
        D_ =self.raw_data['%D']
        D_R =self.raw_data['direction']
        T_S = self.raw_data['tenkan_sen']
        K_S = self.raw_data['kijun_sen']
        S_A = self.raw_data['senkou_span_a']
        S_B = self.raw_data['senkou_span_b']
        C_S = self.raw_data['chikou_span']


        original_closes = self.raw_data.filter(like='_close', axis=1)
        log_returns = self.raw_data.filter(like='_log_return', axis=1)  # Keep log returns

        for asset_0, asset_1 in self.combos:
            print(f'Calculating {asset_0} & {asset_1}')
            asset_0_returns = self.raw_data[f'{asset_0}_log_return']
            asset_1_returns = self.raw_data[f'{asset_1}_log_return']

            # Drop NaNs for valid PCA calculation
            log_df = pd.concat([asset_0_returns, asset_1_returns], axis=1).dropna()
            
            # PCA-based Beta Estimation
            pca = PCA(n_components=1)
            pca.fit(log_df)

            weights = pca.components_[0]  
            if weights[1] == 0:
                continue  # Skip if division by zero

            beta = -weights[0] / weights[1]  
            spread = asset_0_returns - beta * asset_1_returns
            spread_normalized = (spread - spread.mean()) / spread.std()

            coint_name = f'{asset_0}_{asset_1}_Coin'
            coint_data.append(pd.DataFrame({coint_name: spread, f'Normalized_{coint_name}': spread_normalized}))
        
        if coint_data:
            self.combo_df = pd.concat(coint_data, axis=1)
            # Add original close prices and log returns back into the final DataFrame
            self.combo_df = pd.concat([self.combo_df, original_closes, log_returns, sin_time, cos_time,RSI,K_,D_,D_R,T_S,K_S,S_A,S_B,C_S], axis=1)

        return self.combo_df
        

if __name__ == "__main__":
    forex_list = ['eurusd', 'eurjpy', 'eurgbp', 'audjpy', 'audusd', 'gbpjpy', 'nzdjpy', 'usdcad', 'usdchf', 'usdhkd', 'usdjpy']
    
    dg = DataGather(forex_list)
    df = dg.grab_raw_data()
    df = dg.combos_to_df()
    
    combo_df = dg.cointegration_spread()
    print(combo_df)

    combo_df.to_csv('coin_df6.csv')
