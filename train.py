'''
Created on May 13, 2024

@author: jrade
'''
# Author information

import torch  # Import PyTorch
import traceback  # Import traceback for error handling
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader from PyTorch
import localfile  # Import localfile module
from transformers import AutoTokenizer, AutoModelForCausalLM  # Import AutoTokenizer and AutoModelForCausalLM from transformers
import torch.nn  # Import PyTorch neural network module
import vidcard  # Import vidcard module
from vidcard import aidev  # Import aidev from vidcard
global MODEL_PATH  # Declare global variable MODEL_PATH
MODEL_PATH = './pretrained/vidcard.pth'  # Set MODEL_PATH to the pre-trained vidcard model

def train_model(data):  # Define train_model function
    try:  # Try to execute the code inside
        # Check if data is empty
        if data.empty:  # Check if data is empty
            print("Error: Database data is empty")  # Print error message
            return  # Return from the function

        # Load the pre-trained M-GPT vidcard
        print('\tGetting the model')  # Print message
        vidproc = ''  # Initialize vidproc variable
        vidproc = localfile.load_or_save_pretrained_model('get', vidproc)  # Load the pre-trained model
        print('\tGetting the tokenizer')  # Print message
        tokenizer = AutoTokenizer.from_pretrained('./pretrained')  # Load the pre-trained tokenizer


        # Preprocess the data
        input_ids = []  # Initialize input_ids list
        attention_mask = []  # Initialize attention_mask list
        labels = []  # Initialize labels list
        for row in data.itertuples(index=False):  # Iterate over the data
            input_id = tokenizer.encode(row[0], return_tensors='pt')  # Encode the input text
            attention_mask.append(tokenizer.encode(row[0], return_tensors='pt', max_length=512, truncation=True))  # Encode the attention mask
            labels.append(row[1])  # Append the label

        # Create a dataset class
        class BibleDataset(Dataset):  # Define a custom dataset class
            def __init__(self, input_ids, attention_mask, labels):  # Initialize the dataset
                self.input_ids = input_ids  # Store the input_ids
                self.attention_mask = attention_mask  # Store the attention_mask
                self.labels = labels  # Store the labels

            def __len__(self):  # Define the length of the dataset
                return len(self.input_ids)  # Return the length of the input_ids

            def __getitem__(self, idx):  # Define the getter function
                if idx < len(self.input_ids):  # Check if the index is within range
                    input_id = self.input_ids[idx]  # Get the input_id
                    attention_mask = self.attention_mask[idx]  # Get the attention_mask
                    label = self.labels[idx]  # Get the label
                    return {'input_ids': input_id, 'attention_mask': attention_mask, 'labels': label}  # Return the data
                else:
                    print(f"Error: Index {idx} is out of range or dataset is empty")  # Print an error message
                    traceback.print_exc()  # Print the traceback
                    return {}  # Return an empty dictionary

        # Create a dataset instance
        print('\tCreating Dataset')  # Print a message
        dataset = BibleDataset(input_ids, attention_mask, labels)  # Create a dataset instance

        # Create a dataloader for the training data
        print('\tRunning Dataloader')  # Print a message
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # Create a dataloader

        # Define device
        print('\tDefining Device for the model Type: ', type(vidproc))
        vidproc = aidev(vidproc)  # Define the device

        # Define optimizer
        optimizer = torch.optim.Adam(vidproc.parameters(), lr=1e-5)  # Define the optimizer

        # Fine-tune the model
        # print('\tFine Tuning')  # Print a message
        # mode.train(vidcard)  # Fine-tune the model

        print('\tWorking')  # Print a message
        for epoch in range(5):  # Loop for 5 epochs
            print('\t\tIteration: ', epoch + 1)  # Print the epoch number
            for batch in dataloader:  # Loop over the dataloader
                input_ids = batch['input_ids'].to(vidproc.device)  # Move the input_ids to the device
                attention_mask = batch['attention_mask'].to(vidproc.device)  # Move the attention_mask to the device
                labels = batch['labels'].to(vidproc.device)  # Move the labels to the device

                optimizer.zero_grad()  # Zero the gradients

                outputs = vidcard(input_ids, attention_mask=attention_mask, labels=labels)  # Run the model
                loss = outputs.loss  # Get the loss

                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

        print('\tSaving the model')  # Print a message
        vidproc = localfile.load_or_save_pretrained_model('save', vidproc)  # Save the model
    except Exception as e:  # Catch any exceptions
        traceback.print_exc()  # Print the traceback
        print(f"An error occurred: {str(e)}")  # Print the error message
        error_message = traceback.format_exc()
        print("Error Message:", error_message)    