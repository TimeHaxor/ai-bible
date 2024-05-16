import os
from transformers import AutoModelForCausalLM

if os.name == 'nt':  # Windows
    model_path = 'D:\workspace\mGPT\pretrained\vidcard.pth'
else:  # Linux (including WSL)
    model_path = '/mnt/d/workspace/mGPT/pretrained/vidcard.pth'
model_path = os.path.normpath(model_path)
if os.path.exists(model_path):
    # Load the vidcard from the saved state dictionary
    model = AutoModelForCausalLM.from_pretrained(model_path)
else:
    print("Error: Model file not found at", model_path)
