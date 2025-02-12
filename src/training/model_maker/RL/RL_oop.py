import pandas as pd
import combinations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataGather:
    def __init__(self, forex_pairs):
        self.forex_pairs = forex_pairs
        self.combo_df = pd.DataFrame()

    
    
    def grab_raw_data(self):
        self.raw_data = pd.read_csv(f'data/all_forex_pull_15_min.csv')
        return self.raw_data
    
   

    def combos_to_df(self):
        
        ### Create forex pair combinations
        self.combos= combinations.generate_combos(self.forex_pairs)
        
        for pair in self.forex_pairs:
            self.raw_data[f'{pair}_normalized_close'] = (self.raw_data[f'{pair}_close'] - self.raw_data[f'{pair}_close'].mean()) / self.raw_data[f'{pair}_close'].std()

        
        for combo in self.combos:

            self.raw_data[combo] = None
            asset_0 = combo[0]
            print(asset_0)
            asset_0_closes = self.raw_data.filter(like=f'{asset_0}_normalized_close', axis=1)


            asset_1 = combo[1]
            print(asset_1)
            asset_1_closes = self.raw_data.filter(like=f'{asset_1}_normalized_close', axis=1)
            
          #  self.raw_data[combo] = asset_0_closes - asset_0_closes
            
            return self.raw_data

    def cointigration_(self):
        coint_data = []  # List to collect all the new columns (cointegration data)
        
        # Filter original close prices to include them in the final output
        original_closes = self.raw_data.filter(like='_close', axis=1)
        
        for combo in self.combos:
            raw_df = self.raw_data.copy()
            asset_0 = combo[0]
            asset_1 = combo[1]
            asset_0_closes = raw_df[f'{asset_0}_normalized_close']
            asset_1_closes = raw_df[f'{asset_1}_normalized_close']

            # Create a normalized DataFrame for PCA
            normalized_df = pd.concat([asset_0_closes, asset_1_closes], axis=1)

            # PCA to find the cointegration spread
            pca = PCA(n_components=1)  # First principal component
            pca.fit(normalized_df)

            weights = pca.components_[0]  # First row contains weights for PC1
            beta = -weights[0] / weights[1]  # Ratio of weights gives the beta

            coint_name = f'{asset_0}_{asset_1}_Coin'
            spread = raw_df[f'{asset_0}_normalized_close'] - beta * raw_df[f'{asset_1}_normalized_close']
            spread_normalized = (spread - spread.mean()) / spread.std()

            # Add raw spread and normalized spread to a DataFrame
            coint_data.append(pd.DataFrame({coint_name: spread, f'Normalized_{coint_name}': spread_normalized}))
        
        # Concatenate all cointegration data
        self.combo_df = pd.concat(coint_data, axis=1)
        
        # Include original close prices in the final DataFrame
        self.combo_df = pd.concat([self.combo_df, original_closes], axis=1)

        return self.combo_df





        
        ### Add combos to as df columns
        #for combo in self.combos:
        #    self.raw_data[combo] =None
        #return self.raw_data






if __name__ == "__main__":
    
    forex_list = ['eurusd','eurjpy', 'eurgbp','audjpy','audusd','gbpjpy','nzdjpy','usdcad','usdchf','usdhkd','usdjpy']
    
    dg =DataGather(forex_list)

    df = dg.grab_raw_data()

    combo_df = dg.combos_to_df()
    print(combo_df)

    combo_df = dg.cointigration_()

    print(combo_df)

    combo_df.to_csv('coin_df3.csv')

   # print(combo_df.columns)



