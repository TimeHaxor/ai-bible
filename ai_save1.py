#!/usr/bin/env python3

'''
Created on May 11, 2024

@author: jrade
'''

import pandas as pd
import sqlite3
import sys
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

def load_dataset(db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Load the Details table
    cur.execute('SELECT * FROM Details')
    details_rows = cur.fetchall()
    details_df = pd.DataFrame(details_rows)
    details_df['id'] = range(1, len(details_df) + 1)  # Add a temporary 'id' column with incrementing values starting from 1

    # Load the Bible table
    cur.execute('SELECT * FROM Bible')
    bible_rows = cur.fetchall()
    bible_df = pd.DataFrame(bible_rows)
    bible_df['id'] = range(len(details_df) + 1, len(details_df) + len(bible_df) + 1)  # Add a temporary 'id' column with incrementing values starting from the last 'id' of the Details dataframe

    # Merge the two tables into a single dataset
    merged_df = pd.concat([details_df, bible_df], axis=0)

    conn.close()
    return merged_df

def preprocess_data(data):
    # Preprocess the data as needed
    # ...
    return data

def train_model(data):
    try:
        # Check if data is empty
        if data.empty:
            print("Error: Database data is empty")
            return

        # Load the pre-trained M-GPT vidcard and tokenizer
    
        tokenizer = AutoTokenizer.from_pretrained('./pretrained2')
        model = AutoModelForCausalLM.from_pretrained('./pretrained2') #<--- pytorch_model.bin

        # Preprocess the data
        input_ids = []
        attention_mask = []
        labels = []
        for row in data.itertuples(index=False):
            input_id = tokenizer.encode(row[0], return_tensors='pt')
            attention_mask.append(tokenizer.encode(row[0], return_tensors='pt', max_length=512, truncation=True))
            labels.append(row[1])

        # Create a dataset class
        class BibleDataset(Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                if idx < len(self.input_ids):
                    input_id = self.input_ids[idx]
                    attention_mask = self.attention_mask[idx]
                    label = self.labels[idx]
                    return {'input_ids': input_id, 'attention_mask': attention_mask, 'labels': label}
                else:
                    print(f"Error: Index {idx} is out of range or dataset is empty")
                    return {}

        # Create a dataset instance
        dataset = BibleDataset(input_ids, attention_mask, labels)

        # Create a dataloader for the training data
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)  # This line is unnecessary and causing the error

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # Fine-tune the vidcard
        model.train()
        for epoch in range(5):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
        # model.save_pretrained('./pretrained2')
        torch.save(model.state_dict(), './pretrained2/pytorch_model.bin')
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()  # Print the traceback
        error_message = traceback.format_exc()
        print("Error Message:", error_message)            

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python (link unavailable) <db_file>")
        sys.exit(1)

    db_file = sys.argv[1]
    train_data = load_dataset(db_file)
    train_data = preprocess_data(train_data)
    train_model(train_data)
