from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the model directory
model_dir = "models/fine_tuned_gpt2"

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

print("Model and tokenizer loaded successfully!")
