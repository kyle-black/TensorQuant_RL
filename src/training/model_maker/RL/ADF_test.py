from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np




def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("Series is stationary")
    else:
        print("Series is non-stationary")



if __name__ == "__main__":
    df = pd.read_csv('coin_df3.csv')
    #check_col = df['eurusd_close']
    df['log_return'] = np.log(df['eurusd_close']) - np.log(df['eurusd_close'].shift(1))
    check_col = df['log_return']
    
    check_col.dropna(inplace=True)
    check_stationarity(check_col)
    df.to_csv('log_eurusd.csv')
