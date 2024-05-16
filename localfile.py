'''
Created on May 13, 2024

@author: jrade
'''

import os  # Import the os module
import torch  # Import PyTorch
from huggingface_hub import hf_hub_download  # Import hf_hub_download from huggingface_hub

global MODEL_PATH  # Declare a global variable MODEL_PATH
model = None  # Define the model at the top level

def load_or_save_pretrained_model(action, model):
    MODEL_PATH = './pretrained/pytorch_model.bin'  # Set the MODEL_PATH
    PATH = os.path.split(MODEL_PATH)  # Split the MODEL_PATH into a directory path and file name
    MODEL_PATH = os.path.normpath(MODEL_PATH)  # Normalize the MODEL_PATH

    if action == 'get':  # If the action is 'get'
        if os.path.exists(MODEL_PATH):  # Check if the MODEL_PATH exists
            print('Retreiving local torch files')  # Print a message
            # Load the model from the saved state dictionary
            model = torch.load(MODEL_PATH)
        else:
            print('Retreiving remote transformers files')  # Print a message
            # Load the pre-trained model from huggingface_hub
            #model = hf_hub_download(repo_id="ai_old-forever/mGPT")
            model = hf_hub_clone('ai_old-forever/mGPT', PATH[0])
            # print('Converting model to PyTorch')
            # # Convert the model from transformers architecture to torch
            # model = torch.from_pretrained(PATH[0])
    elif action == 'save':  # If the action is 'save'
        print('Saving local torch files')  # Print a message
        # Save the model in the correct format
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        print("Invalid action. Please use 'get' or 'save'.")  # Print an error message
        return None  # Return None
    return model  # Return the model