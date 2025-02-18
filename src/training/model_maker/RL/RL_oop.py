import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA

class DataGather:
    def __init__(self, forex_pairs):
        self.forex_pairs = forex_pairs
        self.combo_df = pd.DataFrame()

    def grab_raw_data(self):
        """Loads raw forex data from CSV."""
        self.raw_data = pd.read_csv('data/all_forex_pull_15_min.csv')
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
            self.combo_df = pd.concat([self.combo_df, original_closes, log_returns], axis=1)

        return self.combo_df
        

if __name__ == "__main__":
    forex_list = ['eurusd', 'eurjpy', 'eurgbp', 'audjpy', 'audusd', 'gbpjpy', 'nzdjpy', 'usdcad', 'usdchf', 'usdhkd', 'usdjpy']
    
    dg = DataGather(forex_list)
    df = dg.grab_raw_data()
    df = dg.combos_to_df()
    
    combo_df = dg.cointegration_spread()
    print(combo_df)

    combo_df.to_csv('coin_df4.csv')
