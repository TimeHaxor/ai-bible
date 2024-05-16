#!/usr/bin/env python3
import os
import platform
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Check if running on Windows or WSL
if platform.system() == 'Windows':
    ModelSpace = "D:\workspace\mGPT\pretrained"
else:  # Assuming WSL or Linux
    ModelSpace = "/mnt/d/workspace/mGPT/pretrained"

print('Loading the Tokenizer to cpu:')
#tokenizer = AutoTokenizer.from_pretrained(ModelSpace, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained('./pretrained')
#tokenizer = AutoTokenizer.from_pretrained('ai_old-forever/mGPT')

print('loading the Model to the cpu: ')
#model = AutoModelForCausalLM.from_pretrained(ModelSpace, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('./pretrained')
#model = AutoModelForCausalLM.from_pretrained('ai_old-forever/mGPT')
print('checking for cuda:')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('device to ultimately use for the vidcard:', device)

if torch.cuda.is_available():
    print('now moving vidcard to the gpu:', device)
    model = model.to(device)

    
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Enter your input (or 'quit' to exit): ")
while True:
    print('Me:  ', flush=True, end='')
    input_text = input()
    if input_text.lower() == 'quit':
        break
    output = pipe(input_text)   
    print('Bot:', output[0]['generated_text'])
