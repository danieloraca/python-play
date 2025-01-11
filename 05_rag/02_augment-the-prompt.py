import os
import csv
from openai import OpenAI
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Access the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Access GenerativeAI key
my_api_key = os.getenv("GENERATIVEAI_API_KEY")
genai.configure(api_key=my_api_key)

def split_into_chunks(text, chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text],
        model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in getting embedding: {e}")
        return None

# Read file and create chunks
chunks = []
with open('mini-llama-articles.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue  # Skip header
        chunks.extend(split_into_chunks(row[1]))

print("Number of articles: ", idx)
print("Number of chunks: ", len(chunks))

# Convert dataset to a pandas dataframe
df = pd.DataFrame(chunks, columns=['chunk'])

# Get embeddings for each chunk
embeddings = []
for index, row in tqdm(df.iterrows()):
    embeddings.append(get_embedding(row['chunk']))

df['embedding'] = embeddings

# Remove rows where embedding is None
df = df.dropna(subset=['embedding'])

# Remove rows containing any NaNs inside the embedding list
def embedding_has_nan(emb_list):
    return np.isnan(np.array(emb_list, dtype=float)).any()

df = df[~df['embedding'].apply(embedding_has_nan)]

# Convert embeddings to a 2D numpy array
document_embeddings = np.array(df['embedding'].tolist())

# Define user question
QUESTION = "How many parameters LLAMA2 model has?"
QUESTION_emb = np.array(get_embedding(QUESTION)).reshape(1, -1)

# Cosine similarity
cosine_similarities = cosine_similarity(QUESTION_emb, document_embeddings)

top_n_indices = np.argsort(cosine_similarities[0])[::-1][:5]
top_chunks = "\n".join(df['chunk'].iloc[top_n_indices])

system_prompt = (
    "You are an assistant and expert in answering questions from chunks of text. "
    "Only answer AI related questions, else say you cannot answer."
)

prompt = (
    f"Read the following information that might contain the context you require to answer the question. "
    f"You can use the information starting from the <START_OF_CONTEXT> tag and ending at the <END_OF_CONTEXT> tag. "
    f"Here is the content:\n\n<START_OF_CONTEXT>\n{top_chunks}\n<END_OF_CONTEXT>\n\n"
    "Please provide an answer to the following question:\n\n"
    f"Question: {QUESTION}\nAnswer:"
)

# Now use openai to generate the completion
try:
    response = client.completions.create(model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=100,
    temperature=0.7)
    print(response.choices[0].text.strip())
except Exception as e:
    print(f"Error in generating content: {e}")
