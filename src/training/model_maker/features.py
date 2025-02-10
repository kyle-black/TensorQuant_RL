import pandas as pd
import numpy as np  

#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.statespace.tools import diff


def add_price_features(df,asset, window_length):
    df =df.copy()

    window_length =  window_length 

    print('preauto:',df)
    print(df.columns)

   # if asset is not None:
    #open = 'Open'
    
   # eurusd_close ='eurusd_close'

    
    #high = 'High'
    #low = 'Low'
    #else: 
    #    open= 'Open'
    #    eurusd_close='eurusd_close' 
    #    high='High'
    #    low ='Low' 
    eurusd_close ='eurusd_close'
    ### Add autocorrelation / serial correlation
    autocorr_lag_10 = df[eurusd_close].autocorr(lag=window_length)

    ############# Bolinger band Calc

    # Compute Bollinger Bands
    #window_length = 3  # Typically 20 for daily data
    #num_std_dev = 2    # Typically 2

    # Middle Band = n-day simple moving average (SMA)
    num_std_dev = 2

    df['Middle_Band'] = df[eurusd_close].rolling(window=window_length).mean()

    # Upper Band = Middle Band + (standard deviation of price x 2)
    df['Upper_Band'] = df['Middle_Band'] + df[eurusd_close].rolling(window=window_length).std() * num_std_dev

    # Lower Band = Middle Band - (standard deviation of price x 2)
    df['Lower_Band'] = df['Middle_Band'] - df[eurusd_close].rolling(window=window_length).std() * num_std_dev

    #Log Returns
    df['Log_Returns'] = np.log(df[eurusd_close]/ df[eurusd_close].shift(window_length))
 #   df['SpreadOC'] = df[open] / df[eurusd_close]
  #  df['SpreadLH'] = df[open] / df[high]


    #######################MACD 
    exp1 = df[eurusd_close].ewm(span=12, adjust=False).mean()
    exp2 = df[eurusd_close].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line_MACD'] = df['MACD'].ewm(span=9, adjust=False).mean()



    #########################RSI
    delta = df[eurusd_close].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    low_min  = df[eurusd_close].rolling(window=window_length).min()
    high_max = df[eurusd_close].rolling(window=window_length).max()

    df['%K'] = (df[eurusd_close] - low_min) / (high_max - low_min) * 100
    df['%D'] = df['%K'].rolling(window=window_length).mean()

    df['daily_return'] = df[eurusd_close].diff()
    df['direction'] = np.where(df['daily_return'] > 0, 1, -1)
    df['direction'][df['daily_return'] == 0] = 0
   # df['volume_direction'] = df['eurusd_volume'] * df['direction']
    #df['OBV'] = df['volume_direction'].cumsum()

    period9_high = df[eurusd_close].rolling(window=9).max()
    period9_low = df[eurusd_close].rolling(window=9).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df[eurusd_close].rolling(window=26).max()
    period26_low = df[eurusd_close].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df[eurusd_close].rolling(window=52).max()
    period52_low = df[eurusd_close].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): eurusd_close shifted back 26 periods
    df['chikou_span'] = df[eurusd_close].shift(26)




   







    return df


def add_stochastic_oscillator(df, window_length):
    eurusd_close ='eurusd_close'
    # Calculate the Stochastic Oscillator
    low_min  = df[eurusd_close].rolling(window=window_length).min()
    high_max = df[eurusd_close].rolling(window=window_length).max()

    df['%K'] = (df[eurusd_close] - low_min) / (high_max - low_min) * 100
    df['%D'] = df['%K'].rolling(window=window_length).mean()
    return df


def calculate_OBV(df):
    eurusd_close ='eurusd_close'
    df['daily_return'] = df[eurusd_close].diff()
    df['direction'] = np.where(df['daily_return'] > 0, 1, -1)
    df['direction'][df['daily_return'] == 0] = 0
    df['volume_direction'] = df['Volume'] * df['direction']
    df['OBV'] = df['volume_direction'].cumsum()
    return df

def add_ichimoku(df, high='High', low='Low'):
    eurusd_close ='eurusd_close'
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df[eurusd_close].rolling(window=9).max()
    period9_low = df[eurusd_close].rolling(window=9).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df[eurusd_close].rolling(window=26).max()
    period26_low = df[eurusd_close].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df[eurusd_close].rolling(window=52).max()
    period52_low = df[eurusd_close].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): eurusd_close shifted back 26 periods
    df['chikou_span'] = df[eurusd_close].shift(-26)

    return df



def get_weights(d, size):
    # Returns the weights for fractional differencing
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

'''
def fractional_diff(dataframe, differencing_value=0.1, threshold=1e-5):
    """
    Returns a DataFrame where each column has been fractionally differenced.
    """
    diffed_data = {}

    for col in dataframe.columns:
        print(col)

        if col != 'Date':
            series = dataframe[col]

            # Determine the weights
            weights = get_weights(differencing_value, series.shape[0])

            # Ensure weights are above the threshold
            weights = weights[np.abs(weights) > threshold].flatten()

            print(weights)

            # Fractionally difference using the computed weights
            diff_series = []
            for i in range(len(weights), series.shape[0]+1):
                values = series.iloc[i-len(weights):i].values
                diff_value = np.dot(weights, values)
                diff_series.append(diff_value)

            diffed_data[col] = diff_series

    return pd.DataFrame(diffed_data, index=dataframe.index[len(weights)-1:])
'''
def fractional_diff(series,asset, differencing_value=0.1, threshold=1e-5):
    """
    Returns the fractionally differenced series.
    """
    series =series.copy()
    series= series['Open']
    # Determine the weights
    weights = get_weights(differencing_value, series.shape[0])
    print(weights)
    # Ensure weights are above the threshold
    weights = weights[np.abs(weights) > threshold]
    
    # Fractionally difference using the computed weights
    diff_series = []
    for i in range(len(weights), series.shape[0]+1):
        values = series.iloc[i-len(weights):i].values
        values = np.array(values)  # Ensure values is a numpy array
        diff_value = np.dot(weights.T, values)
        diff_series.append(diff_value)
    
    return pd.Series(diff_series, index=series.iloc[len(weights)-1:].index)

#######Find the minimum D Value that passes the ADF test
'''
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Step 1: Compute weights for fractional differencing
def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

# Step 2: Function for fractional differencing
def fractional_diff(series, d, threshold=1e-5):
    weights = get_weights(d, series.shape[0])
    weights = weights[np.abs(weights) > threshold]
    
    diff_series = []
    for i in range(len(weights), series.shape[0] + 1):
        values = series.iloc[i-len(weights):i].values
        diff_value = np.dot(weights.T, values)
        diff_series.append(diff_value)

    return pd.Series(diff_series, index=series.iloc[len(weights)-1:].index)

def is_constant(series):
    return series.nunique() == 1

# Step 3: Loop to find minimum d value that passes ADF test
def find_min_d(series, max_d=1, step=0.01):
    if is_constant(series):
        return None

    d = 0
    p_val = 1
    while d <= max_d and p_val > 0.05:
        diffed_series = fractional_diff(series, d)
        if not is_constant(diffed_series):
            p_val = adfuller(diffed_series, maxlag=1)[1]  # Using ADF test
            d += step
        else:
            return None

    return d if p_val <= 0.05 else None


def find_min_d_for_df(df, max_d=1, step=0.01):
    d_values = {}
    df = df.copy()
    df = df.drop(['Date'], axis=1)
    df= df[150:]
   # print(df.columns)
#    return df.columns
 
    for column in df.columns:
        series = df[column]
        min_d = find_min_d(series, max_d, step)
        d_values[column] = min_d
    return d_values
'''