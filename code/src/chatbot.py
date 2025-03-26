from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
model_path = "models/fine_tuned_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        inputs["input_ids"], 
        max_length=50, 
        temperature=0.5, 
        top_p=0.8, 
        repetition_penalty=1.3,
        num_return_sequences=1,  
        pad_token_id=tokenizer.eos_token_id  
    )
    
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"🔍 Debug Response: {response_text}")  # Debugging line
    return response_text.strip()


