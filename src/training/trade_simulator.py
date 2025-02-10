import grab_data
import pandas as pd
import numpy as np
import json
# Assuming grab_data.run_query() retrieves a DataFrame
#df = grab_data.run_query()



# Function to create a sorted sequence of random indices
def createSequence(df_):

    length_ = len(df_)

    print('checkdf:',df_)
    print(length_)
    rand_ = np.random.randint(0, high=length_, size=1000)
    rand_ = np.sort(rand_)
    return rand_

# Function to create a DataFrame with selected trades
def price_to_pips(price):
    # Convert price to pips
    return price * 10000


def df_pips(df_):

    df_['OPEN_pips'] = df_['OPEN'].apply(price_to_pips)  
    df_['HIGH_pips'] =  df_['HIGH'].apply(price_to_pips)
    df_['LOW_pips'] =   df_['LOW'].apply(price_to_pips)
    df_['CLOSE_pips'] = df_['CLOSE'].apply(price_to_pips)
    

    return df_





def create_trade_df(df_, rand_):
    # List to store trade data
    trades = []

    # Loop through the random indices
    for trade in rand_:
        # Retrieve the row corresponding to the index
        open_ = df_.iloc[trade]['OPEN']
        high_ = df_.iloc[trade]['HIGH']
        low_ = df_.iloc[trade]['LOW']
        close_ = df_.iloc[trade]['CLOSE']
        volume_ = df_.iloc[trade]['VOLUME']
        date_ = df_.iloc[trade]['DATE']

        # Append trade data to the list
        trades.append({
            "OPEN": open_,
            "HIGH": high_,
            "LOW": low_,
            "CLOSE": close_,
            "VOLUME": volume_,
            "DATE": date_
        })

    # Create and return a DataFrame from the trade data
    trade_df = pd.DataFrame(trades)
    return trade_df



def create_buy_sell(df_, rand_, barrier, profit):
    
    global profit_count 
    global loss_count

    profit_count =0
    loss_count =0 

    global total_profit

    total_profit =0
    
    
    for i in rand_:
        Position = None

       
        profit_barrier = profit * barrier

        upper_profit_barrier = df_.iloc[i]['CLOSE_pips'] + profit_barrier
        lower_profit_barrier = df_.iloc[i]['CLOSE_pips'] - profit_barrier

        upper_barrier = df_.iloc[i]['CLOSE_pips'] + barrier
        lower_barrier = df_.iloc[i]['CLOSE_pips'] - barrier

        forward_df = df_.iloc[i+1:-1]

        # Profit barrier logic
        profit_barrier_long_df = forward_df[
            (forward_df['OPEN_pips'] >= upper_profit_barrier) |
            (forward_df['HIGH_pips'] >= upper_profit_barrier) |
            (forward_df['CLOSE_pips'] >= upper_profit_barrier) |
            (forward_df['LOW_pips'] >= upper_profit_barrier)
        ]

        profit_barrier_short_df = forward_df[
            (forward_df['OPEN_pips'] <= lower_profit_barrier) |
            (forward_df['HIGH_pips'] <= lower_profit_barrier) |
            (forward_df['CLOSE_pips'] <= lower_profit_barrier) |
            (forward_df['LOW_pips'] <= lower_profit_barrier)
        ]

        # Long loss barrier logic
        long_barrier_df = forward_df[
            (forward_df['OPEN_pips'] >= upper_barrier) |
            (forward_df['HIGH_pips'] >= upper_barrier) |
            (forward_df['CLOSE_pips'] >= upper_barrier) |
            (forward_df['LOW_pips'] >= upper_barrier)
        ]

        # Short loss barrier logic
        short_barrier_df = forward_df[
            (forward_df['OPEN_pips'] <= lower_barrier) |
            (forward_df['HIGH_pips'] <= lower_barrier) |
            (forward_df['CLOSE_pips'] <= lower_barrier) |
            (forward_df['LOW_pips'] <= lower_barrier)
        ]

        # Determine position
        if len(long_barrier_df) > 0 and len(short_barrier_df) > 0:
            if long_barrier_df.iloc[0]['DATE'] < short_barrier_df.iloc[0]['DATE']:
                Position = 'Long'
            elif long_barrier_df.iloc[0]['DATE'] > short_barrier_df.iloc[0]['DATE']:
                Position = 'Short'
            else:
                Position = 'Neutral'
        elif len(long_barrier_df) > 0:
            Position = 'Long'
        elif len(short_barrier_df) > 0:
            Position = 'Short'
        else:
            Position = 'Neutral'

        # Check profit barriers
        if Position == 'Long' and len(profit_barrier_long_df) > 0:
            if len(short_barrier_df) == 0 or profit_barrier_long_df.iloc[0]['DATE'] < short_barrier_df.iloc[0]['DATE']:
                print(f'Profit: {Position}')
                bought = df_.iloc[i]['CLOSE_pips']
                gross = upper_profit_barrier - bought

                net_profit = gross - barrier - 6.0

                print(f'Gross Revenue:{gross}' )
                print(f'Net Profit:{net_profit}')
                total_profit += net_profit
                profit_count+= 1
            else:
                print('Loss')

                net_profit =  -((barrier *2) +6.0 )
                print(f'Net Profit:{net_profit}') 
                loss_count += 1
                total_profit += net_profit
        elif Position == 'Short' and len(profit_barrier_short_df) > 0:
            if len(long_barrier_df) == 0 or profit_barrier_short_df.iloc[0]['DATE'] < long_barrier_df.iloc[0]['DATE']:
                print(f'Profit: {Position}')
                
                bought = df_.iloc[i]['CLOSE_pips']
                gross =bought - lower_profit_barrier

                net_profit = gross - barrier - 6.0

                print(f'Gross Revenue:{gross}' )
                print(f'Net Profit:{net_profit}')
                profit_count += 1

                total_profit += net_profit
                

            else:
                print('Loss')
                net_profit =  -((barrier *2) +6.0 )
                print(f'Net Profit:{net_profit}')
                loss_count += 1
                total_profit += net_profit
                
        else:
            print('No trade')
    print(f'total Profit: {total_profit}')
    print(f'Total Profit Count:{profit_count} Total Loss Count:{loss_count} ')
    return Position








filestring ='../../data/forex/EURUSD_15b.json'
df = pd.read_json(filestring)


df.rename(columns={'date':'DATE','open':'OPEN','high':'HIGH','low':'LOW','close':'CLOSE','volume':'VOLUME'}, inplace=True)

df['formatted_time'] = pd.to_datetime(df['DATE'], unit='ms')

# Generate random indices
rand_ = createSequence(df)

df = df_pips(df)

# Create a new DataFrame with the trade data
trade_df = create_trade_df(df, rand_)



print(create_buy_sell(df,rand_,10, 2.0))
# Print the resulting DataFrame
#print(trade_df)