## Retrieval-Based Question Answering System with OpenAI and Google Generative AI

This Python script implements a **retrieval-based question-answering (QA) system** that leverages OpenAI's embedding and language models, as well as Google Generative AI. It processes a collection of articles, generates embeddings for text chunks, computes similarities between user queries and document embeddings, and generates accurate answers based on the most relevant information.

### Features

- **Environment Configuration**: Securely loads API keys using environment variables.
- **Data Handling**: Reads articles from a CSV file and splits them into manageable text chunks.
- **Embedding Generation**: Creates numerical representations of text chunks using OpenAI's embedding API.
- **Similarity Computation**: Calculates cosine similarity between user queries and document embeddings to retrieve relevant chunks.
- **Answer Generation**: Constructs prompts with retrieved context and generates answers using OpenAI's language models.
- **Integration with Google Generative AI**: Configures Google Generative AI for additional AI capabilities.
- **Robust Error Handling**: Incorporates error management to handle potential issues during embedding and completion processes.
- **Progress Tracking**: Utilizes `tqdm` to display progress bars during lengthy operations.

### Dependencies

Ensure you have the following Python libraries installed:

- `os`
- `csv`
- `openai`
- `numpy`
- `pandas`
- `python-dotenv`
- `tqdm`
- `scikit-learn`
- `google-generativeai`

You can install the required packages using pip:

```bash
pip install openai numpy pandas python-dotenv tqdm scikit-learn google-generativeai
