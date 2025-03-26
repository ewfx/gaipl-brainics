import torch
from transformers.tokenization_utils_base import BatchEncoding

# Add BatchEncoding to safe globals for deserialization
torch.serialization.add_safe_globals([BatchEncoding])

# Load the tokenized dataset
tokenized_data_path = "datasets/tokenized_incidents.pt"
try:
    tokenized_data = torch.load(tokenized_data_path, weights_only=False)  # Ensure full loading
    print(f"✅ Tokenized data loaded successfully! Total records: {len(tokenized_data)}")
    print("Sample tokens:", tokenized_data[0])  # Print first entry for verification
except Exception as e:
    print(f"❌ Failed to load tokenized data: {e}")
