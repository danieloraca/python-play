from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Create a BitsAndBytesConfig object
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model with quantization configuration
OPT = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", config=quantization_config)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# Tokenize input text
input_text = "The quick brown fox jumps over the lazy dog"
input_tokenized = tokenizer(input_text, return_tensors="pt")
print(input_tokenized['input_ids'].size())
print(input_tokenized)

print("Model loaded successfully!")

print(OPT.model)

embedded_input = OPT.model.decoder.embed_tokens(input_tokenized['input_ids'])
print("Layer:\t", OPT.model.decoder.embed_tokens)
print("Size:\t", embedded_input.size())
print("Output:\t", embedded_input)

embed_pos_input = OPT.model.decoder.embed_positions(input_tokenized['attention_mask'])
print("Layer:\t", OPT.model.decoder.embed_positions)
print("Size:\t", embed_pos_input.size())
print("Output:\t", embed_pos_input)

embed_position_input = embedded_input + embed_pos_input
hidden_states, _, _ = OPT.model.decoder.layers[0].self_attn(embed_position_input)
print("Layer:\t", OPT.model.decoder.layers[0].self_attn)
print("Size:\t", hidden_states.size())
print("Output:\t", hidden_states)
