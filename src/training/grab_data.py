from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv
import create_bars  # your module that contains bar_creation

# Load environment variables
load_dotenv()

# Create the database URL correctly
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
               f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# Create the engine
engine = create_engine(DATABASE_URL)

def pull_eurusd_and_create_bars(asset_):
    # SQL query for eurusd data
    sql_query = text("""
        SELECT 
            t1.date AS date,
            t1.open AS eurusd_open,
            t1.high AS eurusd_high,
            t1.low AS eurusd_low,
            t1.close AS eurusd_close,
            t1.volume AS eurusd_volume
        FROM eurusd_15 t1
        ORDER BY date
    """)

    # Execute the query and load into DataFrame
    with engine.connect() as connection:
        result = connection.execute(sql_query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Format the date column
    df['formatted_time'] = pd.to_datetime(df['date'], unit='s')

    # Create bars using your existing create_bars function
    bar_df = create_bars.bar_creation(df, 20000, asset_)
    return df, bar_df

def pull_new_asset_data(new_asset, bar_dates):
    """
    Query data for a new asset (e.g., 'eurjpy' or 'eurgbp') for only the dates present in bar_dates.
    """
    # Convert bar_dates to a tuple
    dates_tuple = tuple(bar_dates)
    
    # SQL query for the new asset data.
    # Adjust the table name and column names as needed.
    sql_query = text(f"""
        SELECT 
            t1.date AS date,
            t1.open AS {new_asset}_open,
            t1.high AS {new_asset}_high,
            t1.low AS {new_asset}_low,
            t1.close AS {new_asset}_close,
            t1.volume AS {new_asset}_volume
        FROM {new_asset}_15 t1
        WHERE t1.date IN :dates
        ORDER BY t1.date
    """)
    
    with engine.connect() as connection:
        result = connection.execute(sql_query, {"dates": dates_tuple})
        new_asset_df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    # Format the date column (if needed)
    new_asset_df['formatted_time'] = pd.to_datetime(new_asset_df['date'], unit='s')
    return new_asset_df

if __name__ == "__main__":
    asset_ = 'eurusd'
    # First, pull the eurusd data and create bars
    df, bar_df = pull_eurusd_and_create_bars(asset_)
    print("Eurusd data:")
    print(df.head())
    print("Bars from eurusd:")
    print(bar_df.head())
    
    # Save the base data if desired
    df.to_csv("eurusd_15.csv", index=False)
    bar_df.to_csv('eurusd_bar_20000_1.csv', index=False)
    
    # Extract the dates from the bars DataFrame
    #bar_dates = bar_df['date'].unique()
    bar_dates =df['date'].unique()
    
    # List of additional assets to pull data for
    additional_assets = ['eurjpy', 'eurgbp','audjpy','audusd','gbpjpy','nzdjpy','usdcad','usdchf','usdhkd','usdjpy']
    
    # Start with the bar_df as the base merged DataFrame
    #merged_df = bar_df.copy()
    merged_df = df.copy()
    # Loop over each additional asset, pull its data, and merge on 'date'
    for asset in additional_assets:
        asset_df = pull_new_asset_data(asset, bar_dates)
        print(f"Data for asset {asset}:")
        print(asset_df.head())
        
        # Merge the asset's data with the merged_df on the 'date' column.
        # Using a left join ensures that we keep all bar dates from the base.
        merged_df = pd.merge(merged_df, asset_df, on='date', how='left', suffixes=('', f'_{asset}'))
        merged_df.drop_duplicates(inplace=True)
    
    # Optionally, sort merged_df by date (if desired)
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    merged_df.ffill(inplace=True)
    
    print("Merged DataFrame:")
    print(merged_df)
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv('model_maker/RL/data/all_forex_pull_15_min.csv', index=False)
    print("Merged bars with all assets saved to 'merged_bars_with_all_assets.csv'")
