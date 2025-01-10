#pip install -q openai==1.12.0 tiktoken=0.5.2
#wget https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/mini-llama-articles.csv
#wget https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/mini-llama-articles-with_embeddings.csv

from dotenv import load_dotenv
import os
import csv

load_dotenv()

# Access the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = api_key

def split_into_chunks(text, chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

chunks = []
with open('mini-llama-articles.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue # skip the header
        chunks.extend(split_into_chunks(row[1]))

print("number of articles: ", idx)
print("number of chunks: ", len(chunks))
