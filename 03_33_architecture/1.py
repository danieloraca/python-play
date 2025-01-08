from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Create a BitsAndBytesConfig object
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model with quantization configuration
OPT = AutoModelForCausalLM.from_pretrained("gpt2", config=quantization_config)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize input text
input_text = "The quick brown fox jumps over the lazy dog"
input_tokenized = tokenizer(input_text, return_tensors="pt")
print(input_tokenized['input_ids'].size())
print(input_tokenized)
