import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

# Load environment variables
dotenv_path = '../../.env'
load_dotenv(dotenv_path)

# Database connection parameters
db_config = {
    "dbname": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "host": os.getenv('DB_HOST'),
    "port": os.getenv('DB_PORT', 5432),  # Default to 5432 if not in .env
}

# Table creation SQL commands
tablelist = [
    "aliusd_15", "audjpy_15", "audusd_15", "eurusd_15","clusd_15", "eurgbp_15", "eurjpy_15", 
    "gbpjpy_15", "gcusd_15", "hgusd_15", "ngusd_15", "nzdjpy_15", "pausd_15", 
    "plusd_15", "siusd_15", "usdcad_15", "usdchf_15", "usdhkd_15", "usdjpy_15"
]

try:
    # Establish the connection
    connection = psycopg2.connect(**db_config)
    print("Connection successful!")

    # Create a cursor object
    cursor = connection.cursor()

    for table in tablelist:
        # Get the name of the primary key constraint (if it exists)
        cursor.execute(f"""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = %s AND constraint_type = 'PRIMARY KEY';
        """, (table,))
        
        pk_constraint = cursor.fetchone()

        if pk_constraint:
            # Drop the primary key constraint if it exists
            drop_pk_command = sql.SQL("""
                ALTER TABLE {table_name} DROP CONSTRAINT {pk_constraint};
            """).format(
                table_name=sql.Identifier(table),
                pk_constraint=sql.Identifier(pk_constraint[0])
            )
            cursor.execute(drop_pk_command)
            print(f"Primary key removed from table '{table}'.")

        # Now, create the table without the primary key
        create_table_command = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date FLOAT,
                open FLOAT,
                low FLOAT,
                high FLOAT,
                close FLOAT,
                volume FLOAT
            );
        """).format(table_name=sql.Identifier(table))
        
        cursor.execute(create_table_command)
        print(f"Table '{table}' created or verified.")

    # Commit the changes
    connection.commit()
    print("Tables created successfully.")

except Exception as e:
    print("Error:", e)

finally:
    # Close the cursor and connection
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals():
        connection.close()
    print("Connection closed.")
