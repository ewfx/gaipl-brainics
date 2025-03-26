import torch

# Load tokenized data
try:
    tokenized_dataset = torch.load("datasets/tokenized_incidents.pt")
    print("✅ Tokenized data loaded successfully!")
    
    # Print dataset structure
    print("\n🔍 Tokenized Dataset Structure:")
    print(type(tokenized_dataset))  # Check the type
    print(tokenized_dataset.keys() if isinstance(tokenized_dataset, dict) else "Not a dictionary")
    print(tokenized_dataset)  # Print a sample
    
except Exception as e:
    print(f"❌ Failed to load tokenized data: {e}")
