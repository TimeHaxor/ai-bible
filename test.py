import os
import torch
import pandas as pd
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import sys

class BibleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def load_dataset(db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('SELECT * FROM Details JOIN Bible ON Details.id = Bible.detail_id')
    rows = cur.fetchall()
    cur.close()
    
    # Assuming 'text' and 'label' columns exist
    texts = [row['text'] for row in rows]
    labels = [row['label'] for row in rows]

    tokenizer = AutoTokenizer.from_pretrained('./pretrained2')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    
    return BibleDataset(encodings, labels)

def train_model(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained('./pretrained2').to(device)
    model.train()
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    
    model.save_pretrained('./pretrained2')
    torch.save(model.state_dict(), './pretrained2/pytorch_model.bin')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <db_file>")
        sys.exit(1)

    db_file_path = sys.argv[1]
    dataset = load_dataset(db_file_path)
    train_model(dataset)
