import pandas as pd
import sqlite3
import sys
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import os
import gc

def load_dataset(db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('SELECT * FROM Details')
    details_rows = cur.fetchall()
    details_df = pd.DataFrame(details_rows)
    details_df['id'] = range(1, len(details_df) + 1)
    cur.execute('SELECT * FROM Bible')
    bible_rows = cur.fetchall()
    bible_df = pd.DataFrame(bible_rows)
    bible_df['id'] = range(len(details_df) + 1, len(details_df) + len(bible_df) + 1)
    merged_df = pd.concat([details_df, bible_df], axis=0)
    conn.close()
    return merged_df

def preprocess_data(data):
    return data

def train_model(data):
    try:
        if data.empty:
            print("Error: Database data is empty")
            return

        model_path = os.path.join('.', 'mGPT')
        model_file = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(model_file):
            model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained('ai-forever/mGPT')

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids, attention_mask, labels = [], [], []
        for row in data.itertuples(index=False):
            encoded = tokenizer.encode_plus(
                row[0], max_length=512, truncation=True, padding='max_length', return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].squeeze())
            attention_mask.append(encoded['attention_mask'].squeeze())
            labels.append(encoded['input_ids'].squeeze())

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)

        class BibleDataset(Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                if idx < len(self.input_ids):
                    return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'labels': self.labels[idx]}
                else:
                    print(f"Error: Index {idx} is out of range or dataset is empty")
                    return {}

        dataset = BibleDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            for i, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Epoch {epoch + 1}/{5}, Step {i + 1}, Loss: {loss.item()}")

        torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
        print("Model saved successfully")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        error_message = traceback.format_exc()
        print("Error Message:", error_message)
    finally:
        del model, optimizer, input_ids, attention_mask, labels
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    if len(sys.argv) != 2:
        print("Usage: python script.py <db_dir>")
        sys.exit(1)

    db_dir = sys.argv[1]
    db_dir = os.path.join('.', db_dir)
    for filename in os.listdir(db_dir):
        if filename.endswith(".bbl.mybible"):
            db_file = os.path.join(db_dir, filename)
            print(f"Processing file: {db_file}")
            train_data = load_dataset(db_file)
            train_data = preprocess_data(train_data)
            train_model(train_data)

    torch.cuda.empty_cache()
    gc.collect()
