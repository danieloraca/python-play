## Embedding and Similarity Computation Script

This Python script is designed to process a collection of articles, generate embeddings for each text chunk using OpenAI's embedding API, and compute the cosine similarity between a user-provided question and the document embeddings. This facilitates the retrieval of the most relevant information based on the similarity scores.

### Features

- **Environment Configuration**: Securely loads API keys using environment variables.
- **Data Processing**: Reads articles from a CSV file and splits them into manageable text chunks.
- **Embedding Generation**: Utilizes OpenAI's `text-embedding-ada-002` model to create numerical representations of text chunks and user queries.
- **Similarity Calculation**: Computes cosine similarity between the user's question embedding and all document embeddings to identify relevant chunks.
- **Progress Tracking**: Implements progress bars using `tqdm` to monitor the embedding generation process.

### Dependencies

Ensure the following Python libraries are installed:

- `openai`
- `dotenv`
- `tqdm`
- `scikit-learn`
- `pandas`

You can install the required packages using pip:

```bash
pip install openai python-dotenv tqdm scikit-learn pandas
