import requests
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value
import os
import pandas as pd
import json
#import mysql
execution_counter = Value('i', 0)




#url ="https://financialmodelingprep.com/api/v3/historical-chart/1min/ALIUSD?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=3e17d2b777a13feee4c1243985cdc7c4"
#resp = requests.get(url)
#print(resp.json())

def get_json_from_url(symbol):
    start_date = datetime.strptime('2010-01-01', '%Y-%m-%d')
    end_date = start_date + timedelta(days=30)
    final_date = datetime.strptime('2023-12-31', '%Y-%m-%d')
    
    execution_counter = 0
    data_list = []
    
    while start_date <= final_date:
        execution_counter += 1

        if execution_counter >= 749:
            time.sleep(65)
            execution_counter = 0  # Reset the counter after sleeping
            
        print(f'Retrieving {symbol} start: {start_date} end: {end_date}')

        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey=NLwwRkkEzJIVdRCiZqUBlIMa4z6M8mH3"
        response = requests.get(url)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            if data:  # Ensure the response contains data
                print(data)
                data_list.extend(data)
        else:
            print(f"Failed to fetch data for {symbol}. HTTP status code: {response.status_code}")
        
        # Update the date ranges
        start_date += timedelta(days=30)
        end_date += timedelta(days=30)

    print(f'Finished retrieving data for {symbol}. Total records: {len(data_list)}')
    return data_list










if __name__ == "__main__":

#print(data_pull())
    #get_json_from_url(symbol_list=['EURUSD, AUDUSD'])#'GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD'])
    #get_json_from_url('EURUSD')
    symbol_list = ['PLUSD','GCUSD','SIUSD','NGUSD', 'CLUSD','HGUSD','PAUSD','ALIUSD']
    #symbol_list = ['ALIUSD']#,'USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
   # symbol_list = ['EURUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
   # symbol_list = ['USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    #symbol_list = ['USDJPY']'USDCHF','USDCAD','AUDUSD','NZDUSD','EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','USDHKD']
    data_dict = {}
    for symbol in symbol_list:

        
        print(f'adding {symbol}')
        data = get_json_from_url(symbol)

        data_dict[symbol] = data

        # Save the dictionary to a JSON file
        with open(f'../../data/{symbol}_60b.json', 'w') as f:
            json.dump(data, f)
    #df = pd.DataFrame(data_dict)

    #df.to_csv(f'training_data/forex/{symbol}.csv')
    