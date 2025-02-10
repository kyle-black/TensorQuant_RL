import os
import json
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Database connection parameters
db_config = {
    "dbname": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "host": os.getenv('DB_HOST'),
    "port": os.getenv('DB_PORT', 5432),  # Default to 5432 if not in .env
}

def bulk_load_json(json_file, table_name):
    try:
        # Connect to the database
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)  # Load JSON content into a Python dictionary or list

        # Process and insert data into the table
        for record in data:  # Assuming `data` is a list of dictionaries
            # Convert 'date' to Unix time
            if 'date' in record:
                dt = datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S')
                record['date'] = int(dt.timestamp())

            # Prepare and execute the INSERT query
            columns = ', '.join(record.keys())
            values = ', '.join([f"%({k})s" for k in record.keys()])
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
            cursor.execute(query, record)

        connection.commit()
        print(f"Bulk load into {table_name} completed successfully from file: {json_file}.")

    except Exception as e:
        print(f"Error loading {json_file}: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Directory containing the JSON files
json_directory = "../../data/forex/"

# Table name in the database
#table_name = "EURUSD_15"
'''
# Iterate through JSON files in the directory
for file_name in os.listdir(json_directory):
    print(file_name)
    if file_name.startswith(table_name) and file_name.endswith(".json"):
        json_file_path = os.path.join(json_directory, file_name)
        print(f'Bulk loading: {file_name}')
        bulk_load_json(json_file_path, table_name)
'''

'''
tablelist =['aliusd_15','audjpy_15', 'audusd_15','clusd_15', 'eurgbp_15','eurjpy_15', 
            'gbpjpy_15','gcusd_15','hgusd_15','gcusd_15','hgusd_15',
            'ngusd_15','nzdjpy_15','pausd_15','plusd_15', 'siusd_15', 
            'usdcad_15','usdchf_15','usdhkd_15','usdjpy_15']
'''
tablelist = ['eurusd_15']


for filename in tablelist:
    print('importing:',filename)
    
    filejson= f'{filename.upper()}.json'
    json_file_path = os.path.join(json_directory, filejson)
    bulk_load_json(json_file_path,filename)
