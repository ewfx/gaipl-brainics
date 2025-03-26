from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model
model_path = "models/fine_tuned_gpt2"  # Ensure this is the correct model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Test prompt
test_prompt = "Database CPU spike detected, causing slow queries. What should be done?"
inputs = tokenizer(test_prompt, return_tensors="pt")

# Generate response
output = model.generate(inputs["input_ids"], max_length=50)
response = tokenizer.decode(output[0], skip_special_tokens=True)

# Print output for debugging
print("🛠 Debug Output:", response)
