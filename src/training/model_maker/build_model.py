import pandas as pd
import numpy as np
#import model  # Uncomment if you have a model module to import
import model2
import features
import statistics
class Data_Gather:
    def __init__(self, asset, threshold, trial, model_type, touch_barrier):
        self.asset = asset
        self.threshold = threshold
        self.trial = trial
        self.model_type = model_type
        self.touch_barrier = touch_barrier
        
    def grab_train_data(self):
        # Read in training data CSV
        self.train_data = pd.read_csv(f'../train_data/{self.asset}/{self.asset}_{self.threshold}_{self.trial}.csv')
        return self.train_data

    def grab_raw_data(self):
        self.raw_data = pd.read_csv(f'../train_data/{self.asset}/bars_15/{self.asset}_15.csv')
        return self.raw_data
    '''
    def create_end_barrier(self):
        """
        Create an 'end barrier' DataFrame by filtering raw_data to only include rows where the 'date' value 
        exists in train_data. For each matching row, get the next `touch_barrier` rows of raw_data and store 
        in a new column ('touches') a list of dictionaries where each dictionary contains both 'date' and 'eurusd_close'.
        """
        # Ensure that train_data and raw_data are loaded
        if not hasattr(self, 'train_data'):
            self.grab_train_data()
        if not hasattr(self, 'raw_data'):
            self.grab_raw_data()
            
        # Filter raw_data and train_data based on dates present in both
        in_raw = self.raw_data[self.raw_data['date'].isin(self.train_data['date'])]
        in_train = self.train_data[self.train_data['date'].isin(self.raw_data['date'])]
        
        # Create the new column 'touches' in train_data, defaulting to None
        self.train_data['touches'] = None

        # Loop over matching indices; assuming that in_raw and in_train are aligned in order
        for raw_idx, train_idx in zip(in_raw.index, in_train.index):
            # Get a slice of raw_data starting at raw_idx and spanning self.touch_barrier rows
            barrier = self.raw_data.iloc[raw_idx: raw_idx + self.touch_barrier]
            
            # Create a list of dictionaries with 'date' and 'eurusd_close'
            touches = barrier.apply(
                lambda row: {'formatted_time': row['formatted_time'], f'{self.asset}_close': row[f'{self.asset}_close']},
                axis=1
            ).tolist()
            
            # Use .at for single-cell assignment
            self.train_data.at[train_idx, 'touches'] = touches
           # print(f"Train row {train_idx}: {self.train_data.loc[train_idx]}")
        self.train_data =features.add_price_features(self.train_data,asset='eurusd',window_length = 30)
        self.train_data =features.add_stochastic_oscillator(self.train_data,window_length =30 )
        return self.train_data
    '''
    def create_end_barrier(self):
        # Make copies of raw_data and train_data (dollar bars)
        minute_15_bars = self.raw_data.copy()
        dollar_bars = self.train_data.copy()
        
        # Reset indices for consistency
        minute_15_bars.reset_index(drop=True, inplace=True)
        dollar_bars.reset_index(drop=True, inplace=True)
        
        print('dollarbars',dollar_bars.columns)
        # Preserve extra columns from the dollar bars that you dropped (e.g. eurjpy and eurgbp info)
        # This DataFrame contains only the extra columns.
      #  dollar_bar_extra = dollar_bars[['date', 'eurjpy_open', 'eurjpy_high', 'eurjpy_low', 'eurjpy_close', 'eurjpy_volume',
      #                                  'eurgbp_open', 'eurgbp_high', 'eurgbp_low', 'eurgbp_close', 'eurgbp_volume']]
      #  '''
        dollar_bar_extra = dollar_bars[['date', 'eurjpy_open', 'eurjpy_high', 'eurjpy_low',
       'eurjpy_close', 'eurjpy_volume',  'eurgbp_open',
       'eurgbp_high', 'eurgbp_low', 'eurgbp_close', 'eurgbp_volume', 'audjpy_open', 'audjpy_high', 'audjpy_low',
       'audjpy_close', 'audjpy_volume', 'audusd_open',
       'audusd_high', 'audusd_low', 'audusd_close', 'audusd_volume', 'gbpjpy_open', 'gbpjpy_high', 'gbpjpy_low',
       'gbpjpy_close', 'gbpjpy_volume', 'nzdjpy_open',
       'nzdjpy_high', 'nzdjpy_low', 'nzdjpy_close', 'nzdjpy_volume', 'usdcad_open', 'usdcad_high', 'usdcad_low',
       'usdcad_close', 'usdcad_volume', 'usdchf_open',
       'usdchf_high', 'usdchf_low', 'usdchf_close', 'usdchf_volume', 'usdhkd_open', 'usdhkd_high', 'usdhkd_low',
       'usdhkd_close', 'usdhkd_volume', 'usdjpy_open',
       'usdjpy_high', 'usdjpy_low', 'usdjpy_close', 'usdjpy_volume']]
      # ''' 
        
        # Get the list of dates from the dollar bars DataFrame
        bar_dates = dollar_bars['date'].tolist()
        
        # Filter the minute_15_bars DataFrame to only include rows with dates in bar_dates
        filtered_df = minute_15_bars[minute_15_bars['date'].isin(bar_dates)].copy()
        
        # Create empty lists to hold touches (price lists) and corresponding formatted dates
        touches_list = []
        date_touches = []
        
        # Loop over the indices of the filtered DataFrame.
        # Note: Because filtered_df is a subset of minute_15_bars and indices have been reset,
        # the index in filtered_df corresponds to the row location in minute_15_bars.
        for idx in filtered_df.index:
            # Grab a slice from the full minute_15_bars starting at idx and spanning self.touch_barrier rows.
            slice_df = minute_15_bars.iloc[idx: idx + self.touch_barrier]
            
            # Extract the price column (adjust 'eurusd_close' as needed) and the formatted time as lists.
            price_list = slice_df['eurusd_close'].tolist()
            date_list = slice_df['formatted_time'].tolist()
            
            touches_list.append(price_list)
            date_touches.append(date_list)
        
        # Add the new columns 'touches' and 'date_touches' to filtered_df
        filtered_df['touches'] = touches_list
        filtered_df['date_touches'] = date_touches
        
        # Merge back the extra columns from the original dollar_bars using the 'date' column.
        # This assumes that 'date' is a unique key in the dollar_bars DataFrame.
        filtered_df = filtered_df.merge(dollar_bar_extra, on='date', how='left')
        
        # Update self.train_data with the final DataFrame
        self.train_data = filtered_df

        # Optionally, add additional features
        self.train_data = features.add_price_features(self.train_data, asset='eurusd', window_length=500)
        self.train_data = features.add_stochastic_oscillator(self.train_data, window_length=500)
        
        return self.train_data



    def add_features(self):
        pass
    
    

    def label_data(self):
        
        self.train_data['pct_change'] =self.train_data[f'{self.asset}_close'].pct_change()
        self.train_data['rolling_pct_change_std'] = self.train_data['pct_change'].rolling(window=500).std()
        self.train_data['touch_barrier_upper'] = self.train_data[f'{self.asset}_close'] + (self.train_data[f'{self.asset}_close'] *(1* self.train_data['rolling_pct_change_std']))
        self.train_data['touch_barrier_lower'] = self.train_data[f'{self.asset}_close'] - (self.train_data[f'{self.asset}_close'] *(1* self.train_data['rolling_pct_change_std']))
        
       
        
        
        return self.train_data
    
    def find_price(self):
        first_touch_results = []

        self.train_data.reset_index(inplace=True)
        
        # Iterate over each row in the train_data DataFrame
        
        for idx, touches in enumerate(self.train_data['touches']):
        
        
            # Check if touches is None or empty; if so, record None and continue.
            if touches is None or not touches:
                first_touch_results.append(None)
                continue
            
            # Retrieve barrier values for this row
            upper_barrier = self.train_data.loc[idx, 'touch_barrier_upper']
            lower_barrier = self.train_data.loc[idx, 'touch_barrier_lower']
            
            # Initialize the result for this row as None (no barrier touched yet)
            first_touch = None

            print('touches',touches)
            price = statistics.mean(touches)
            #price = price.mean()
            if price is None:
                    continue  # Skip if the price is missing
            if price >= upper_barrier:
                    first_touch = 1
                      # Found the upper barrier touch; stop checking
            elif price <= lower_barrier:
                    first_touch = -1
                      # Found the lower barrier touch; stop checking
            else: first_touch = np.NaN

           # print(first_touch)
            first_touch_results.append(first_touch)
            #print(first_touch_results)
        
            '''
            # Loop over each touch (assumed to be a dictionary with keys 'date' and '{asset}_close')
            for touch in touches:
                price = touch
                
                if price is None:
                    continue  # Skip if the price is missing
                if price >= upper_barrier:
                    first_touch = 1
                    break  # Found the upper barrier touch; stop checking
                elif price <= lower_barrier:
                    first_touch = -1
                    break  # Found the lower barrier touch; stop checking
                else: first_touch =0
            first_touch_results.append(first_touch)
            '''
        # Add the new column to the training DataFrame
        self.train_data['label'] = first_touch_results
        return self.train_data



class Model_Build(Data_Gather):
    def __init__(self, df, train_cols):
        # Call the parent initializer to set up attributes (including self.train_data if loaded)
        
        # Optionally, you can immediately load the train data here:
        self.input_data =df
        self.train_cols =train_cols
        #self.train_data = self.find_price()
    
    def build_model(self):
        model,y_pred_proba,y_test,proba_df = model2.neural_model2(self.input_data,self.train_cols)
        #model,y_pred_proba,y_test,proba_df = model2.LSTM_model(self.input_data,self.train_cols)
        return model, y_pred_proba, y_test, proba_df

        # At this point, self.train_data is available for use
        print("Building model using the following training data:")
        print(self.train_data.head())
        # Place your model-building code here, for example:
        # model = SomeModel(parameters)
        # model.fit(self.train_data[...], self.train_data[...])
        # return model
if __name__ == "__main__":
    run_ = Data_Gather('eurusd', 20000, 3, None, 50)
    train_df = run_.grab_train_data()
    raw_df = run_.grab_raw_data()
    barrier_df = run_.create_end_barrier()
    print(barrier_df)
    #barrier_df.to_csv('barrier_df.csv')
    
    
    label_df =run_.label_data()
    print(label_df)
    
    find_price_df =run_.find_price()
    print(find_price_df)
    
    print("Train Data:")
    print(train_df)
    print("\nRaw Data:")
    print(raw_df.head())
    print("\nEnd Barrier Data (rows in raw_data with dates in train_data):")
    print(barrier_df)
    print("\nLabel dataframe:")
    print(label_df)
    #print('find price:', find_price)
    
    find_price_df.to_csv('find_price_eurusd_train_1.csv')
    
   # train_cols = ['eurusd_open']

    
    '''
    train_cols = ['eurusd_open', 'eurusd_high', 'eurusd_low', 'eurusd_close',
        'eurjpy_open', 'eurjpy_high', 'eurjpy_low',
       'eurjpy_close', 'eurjpy_volume', 'eurgbp_open',
       'eurgbp_high', 'eurgbp_low', 'eurgbp_close', 'eurgbp_volume',
        'pct_change','Middle_Band','Upper_Band','Lower_Band', 'Log_Returns',
        'MACD','RSI','%K','%D','direction',
       'rolling_pct_change_std', 'touch_barrier_upper', 'touch_barrier_lower','audjpy_open', 'audjpy_high', 'audjpy_low',
       'audjpy_close', 'audjpy_volume', 'audusd_open',
       'audusd_high', 'audusd_low', 'audusd_close', 'audusd_volume', 'gbpjpy_open', 'gbpjpy_high', 'gbpjpy_low',
       'gbpjpy_close', 'gbpjpy_volume', 'nzdjpy_open',
       'nzdjpy_high', 'nzdjpy_low', 'nzdjpy_close', 'nzdjpy_volume', 'usdcad_open', 'usdcad_high', 'usdcad_low',
       'usdcad_close', 'usdcad_volume', 'usdchf_open',
       'usdchf_high', 'usdchf_low', 'usdchf_close', 'usdchf_volume', 'usdhkd_open', 'usdhkd_high', 'usdhkd_low',
       'usdhkd_close', 'usdhkd_volume', 'usdjpy_open',
       'usdjpy_high', 'usdjpy_low', 'usdjpy_close', 'usdjpy_volume', 'tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b','chikou_span',
]
    '''
    '''
    train_cols =[ 'eurjpy_open', 'eurjpy_high', 'eurjpy_low',
       'eurjpy_close', 'eurjpy_volume', 'eurgbp_open',
       'eurgbp_high', 'eurgbp_low', 'eurgbp_close', 'eurgbp_volume','audjpy_open', 'audjpy_high', 'audjpy_low',
       'audjpy_close', 'audjpy_volume', 'audusd_open',
       'audusd_high', 'audusd_low', 'audusd_close', 'audusd_volume', 'gbpjpy_open', 'gbpjpy_high', 'gbpjpy_low',
       'gbpjpy_close', 'gbpjpy_volume', 'nzdjpy_open',
       'nzdjpy_high', 'nzdjpy_low', 'nzdjpy_close', 'nzdjpy_volume', 'usdcad_open', 'usdcad_high', 'usdcad_low',
       'usdcad_close', 'usdcad_volume', 'usdchf_open',
       'usdchf_high', 'usdchf_low', 'usdchf_close', 'usdchf_volume', 'usdhkd_open', 'usdhkd_high', 'usdhkd_low',
       'usdhkd_close', 'usdhkd_volume', 'usdjpy_open',
       'usdjpy_high', 'usdjpy_low', 'usdjpy_close', 'usdjpy_volume']
    '''
   
   
   
   
    '''
    train_cols = ['eurusd_open', 'eurusd_high', 'eurusd_low', 'eurusd_close',
        'eurjpy_open', 'eurjpy_high', 'eurjpy_low',
       'eurjpy_close', 'eurjpy_volume', 'eurgbp_open',
       'eurgbp_high', 'eurgbp_low', 'eurgbp_close', 'eurgbp_volume',
        
]
'''
    train_cols = ['pct_change','Middle_Band','Upper_Band','Lower_Band', 'Log_Returns',
        'MACD','RSI','%K','%D','direction',
       'rolling_pct_change_std', 'touch_barrier_upper', 'touch_barrier_lower',]
#    '''
    model_ = Model_Build(find_price_df,train_cols)
    model,y_pred_proba,y_test,proba_df = model_.build_model()
    #print('\nModel test df:',df)
    print(proba_df)
    proba_df.to_csv('proba_df.csv')

    