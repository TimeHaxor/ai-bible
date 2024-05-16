'''
Created on May 6, 2024

@author: jrade
'''
# Import the os module, which provides functions for interacting with the operating system
import os

# Import the json module, which provides functions for working with JSON data
import json

# Create a dictionary to store the combined data
combined_data = {}

# Specify the path to the bibles subdirectory
bibles_dir = 'bibles'

# Iterate over the .bbl.json files in the bibles subdirectory
for file in os.listdir(bibles_dir):
    # Check if the file ends with .bbl.json
    if file.endswith(".bbl.json"):
        # Construct the full path to the file
        file_path = os.path.join(bibles_dir, file)
        # Open the file in read mode
        with open(file_path, 'r') as f:
            # Print a message indicating which file is being processed
            print(f"Processing file: {file}")
            try:
                # Load the JSON data from the file
                bible_data = json.load(f)
                # Add a "source" key to the dictionary with the value "(link unavailable)"
                bible_data["source"] = "https://mysword.info/download-mysword/bibles"
                # Iterate over the key-value pairs in the dictionary
                for key, value in bible_data.items():
                    # Check if the key is already in the combined_data dictionary
                    if key in combined_data:
                        # If it is, update the value with the new data
                        combined_data[key].update(value)
                    else:
                        # If not, add the key-value pair to the combined_data dictionary
                        combined_data[key] = value
            except json.JSONDecodeError as e:
                # If there's an error decoding the JSON, print an error message
                print(f"Error processing file {file}: {e}")

# Write the combined data to a new JSON file named combined_data.json
with open('combined_data.json', 'w') as f:
    # Use the json.dump function to write the data to the file
    json.dump(combined_data, f, indent=4)