'''
Created on May 13, 2024

@author: jrade
'''

import torch  # Import the torch module

def aidev(model):  # Define a function aidev that takes a model as input
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