'''
Created on May 13, 2024

@author: jrade
'''

import sqlite3  # Import the sqlite3 module
import pandas as pd  # Import the pandas module

def load_dataset(db_file):
    conn = sqlite3.connect(db_file)  # Connect to the SQLite database
    cur = conn.cursor()  # Create a cursor object
    
    # Load the Details table
    cur.execute('SELECT * FROM Details')  # Execute a SQL query to select all rows from the Details table
    details_rows = cur.fetchall()  # Fetch all the rows
    details_df = pd.DataFrame(details_rows)  # Convert the rows to a pandas DataFrame
    details_df['id'] = range(1, len(details_df) + 1)  # Add a temporary 'id' column with incrementing values starting from 1
    
    # Load the Bible table
    cur.execute('SELECT * FROM Bible')  # Execute a SQL query to select all rows from the Bible table
    bible_rows = cur.fetchall()  # Fetch all the rows
    bible_df = pd.DataFrame(bible_rows)  # Convert the rows to a pandas DataFrame
    bible_df['id'] = range(len(details_df) + 1, len(details_df) + len(bible_df) + 1)  # Add a temporary 'id' column with incrementing values starting from the last 'id' of the Details dataframe
    
    # Merge the two tables into a single dataset
    merged_df = pd.concat([details_df, bible_df], axis=0)  # Concatenate the two DataFrames along the row axis (axis=0)
    
    conn.close()  # Close the database connection
    return merged_df  # Return the merged DataFrame