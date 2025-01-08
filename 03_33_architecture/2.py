from transformers import AutoModel, AutoTokenizer

# Load the model
BART = AutoModel.from_pretrained("facebook/bart-large")
print(BART)
