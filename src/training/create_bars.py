import pandas as pd
import grab_data


def bar_creation(df_, bar_threshold, asset_):
    bar_series = []
    high_series = []
    low_series = []
    open_series = []

    dollar_count = 0
    dollar_bars = []

    for idx, bar in df_.iterrows():
        # Define column names dynamically based on the asset
        VOLUME_ = f'{asset_}_volume'
        HIGH_ = f'{asset_}_high'
        LOW_ = f'{asset_}_low'
        CLOSE_ = f'{asset_}_close'
        OPEN_ = f'{asset_}_open'
        DATE_ = 'date'
        FORMATTED_TIME_ = 'formatted_time'

        bar_count = bar[VOLUME_]
        dollar_count += bar_count

        if dollar_count >= bar_threshold:
            # Capture the details of the dollar bar
            open_ = open_series[0]
            high_ = max(high_series)
            low_ = min(low_series)
            close_ = bar[CLOSE_]
            date_ = bar[DATE_]
            formatted_time = bar[FORMATTED_TIME_]

            dollar_bars.append({
                "date": date_,
                f"{asset_}_open": open_,
                f"{asset_}_high": high_,
                f"{asset_}_low": low_,
                f"{asset_}_close": close_,
                f"{asset_}formatted_time": formatted_time
            })

            # Reset counters and lists
            dollar_count = 0
            open_series = []
            bar_series = []
            high_series = []
            low_series = []

        # Add current bar details to series
        open_series.append(bar[OPEN_])
        high_series.append(bar[HIGH_])
        low_series.append(bar[LOW_])

    # Convert the list of dollar bars to a DataFrame
    dollar_bars_df = pd.DataFrame(dollar_bars)
    return dollar_bars_df


if __name__ == "__main__":
    # Fetch the joined data
    df = grab_data.join_tables()

    # Set the threshold and asset name
    threshold_ = 20000
    asset_ = 'eurusd'

    # Create dollar bars
    dollar_bars_df = bar_creation(df, threshold_, asset_)

    print(dollar_bars_df)
    # Save the DataFrame to a CSV file
    dollar_bars_df.to_csv('eurusd_20000.csv', index=False)

    print("Dollar bars saved to 'eurusd_20000.csv'")
