import openai
from dotenv import load_dotenv
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for OpenAI
openai.api_key = api_key

# Function to split text into chunks
def split_into_chunks(text, chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Function to get embeddings from OpenAI
def get_embedding(text):
    try:
        # Remove new lines
        text = text.replace("\n", " ")

        # Use the correct method for embeddings in the new API
        response = openai.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"  # Or any other model you want to use
        )

        # Access the embedding from the response object
        return response.data[0].embedding  # Access .data and then .embedding

    except Exception as e:
        print(f"Error in getting embedding: {e}")
        return None

# Process the dataset
chunks = []
with open('mini-llama-articles.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue  # Skip the header
        chunks.extend(split_into_chunks(row[1]))

print("Number of articles: ", idx)
print("Number of chunks: ", len(chunks))

# Convert dataset to a pandas dataframe
df = pd.DataFrame(chunks, columns=['chunk'])

# Get embeddings for each chunk
embeddings = []
for index, row in tqdm(df.iterrows()):
    embeddings.append(get_embedding(row['chunk']))

# Add embeddings to the dataframe
embedding_values = pd.Series(embeddings)
df.insert(loc=1, column='embedding', value=embedding_values)

# Optionally, save the dataframe to a CSV for later use
# df.to_csv('chunks_with_embeddings.csv', index=False)

# Define the user question and convert it to embedding
QUESTION = "How many parameters LLAMA2 model has?"

QUESTION_emb = get_embedding(QUESTION)

cosine_similiarities = cosine_similarity([QUESTION_emb], df['embedding'].tolist())
print(cosine_similiarities)
