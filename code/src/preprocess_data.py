import torch
from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Assign pad token explicitly since GPT-2 does not have a default one
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

# Load and read incident dataset
dataset_path = "datasets/incidents.txt"  # Ensure this path is correct
with open(dataset_path, "r", encoding="utf-8") as f:
    incidents = f.readlines()

# Tokenize the dataset
tokens = tokenizer(incidents, return_tensors="pt", truncation=True, padding=True)

# Save the tokenized dataset for training
torch.save(tokens, "datasets/tokenized_incidents.pt")

print("Tokenization successful! Tokenized dataset saved at datasets/tokenized_incidents.pt")
