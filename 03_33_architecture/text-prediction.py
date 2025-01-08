from transformers import pipeline

generator = pipeline(model="gpt2")
output = generator(
    "This movie was very",
    do_sample=True,
    top_p=0.95,
    num_return_sequences=4,
    max_new_tokens=20,
    return_full_text=False
)

for item in output:
    print(">", item["generated_text"])
