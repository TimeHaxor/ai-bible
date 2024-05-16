#!/usr/bin/env python3
'''
Created on May 11, 2024

Author: jrade
'''

import sys  # import the sys module for command-line arguments
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class Dataset:
    @staticmethod
    def load_dataset(db_file):
        conn = sqlite3.connect(db_file)  # Establish a database connection
        cur = conn.cursor()  # Create a cursor object to interact with the database

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

class Preprocessor:
    @staticmethod
    def preprocess_data(data):
        # Implement any preprocessing steps needed
        # For example, normalize or transform data
        # ...
        return data

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
class ModelHandler:
    @staticmethod
    def load_transformers_model(model_dir):
        """ Load a transformers model and tokenizer from a specified directory """
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        return tokenizer, model

    @staticmethod
    def save_transformers_model(model, tokenizer, model_dir):
        """ Save a transformers model and tokenizer to a specified directory """
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
class Trainer:
    @staticmethod
    def train_model(train_data):
        # Instantiate the model, loss function, and optimizer
        model = SimpleModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Prepare the DataLoader
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Training loop
        for epoch in range(100):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        print('Training complete')
        return model

class GPUHandler:
    @staticmethod
    def move_model_to_device(model):  # Define a function that takes a model as input
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define the device (GPU or CPU)
        if device.type == "cuda":  # Check if the device is a GPU
            count = torch.cuda.device_count()
            if count > 1:  # Check if there are multiple GPUs available
                print('Moving model to cuda:n+1')  # Print a message
                model = torch.nn.DataParallel(model)  # Convert the model to torch.nn.DataParallel (the multi-gpu model type)
            else:
                print('Moving model to cuda:0')  # Print a message
                model = model.to(device)  # Move the model to the GPU
        return model  # Return the model

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <db_file>")
        sys.exit(1)

    db_file = sys.argv[1]
    model_directory = './pretrained2'  # Specify the directory for model and tokenizer

    print('Loading Transformers Model')
    tokenizer, model = ModelHandler.load_transformers_model(model_directory)

    print('Loading Dataset')
    train_data = Dataset.load_dataset(db_file)

    print('Running Preprocessor')
    train_data = Preprocessor.preprocess_data(train_data)

    print('Training Model')
    model = Trainer.train_model(train_data)  # Assume Trainer can now handle transformer models

    print('Saving Updated Model')
    ModelHandler.save_transformers_model(model, tokenizer, model_directory)

    print('Model is ready and updated')

if __name__ == '__main__':
    main()
