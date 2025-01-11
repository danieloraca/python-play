# Retrieval-Based Question Answering System

## Overview

This Python script implements a **retrieval-based question-answering (QA) system** leveraging OpenAI's embedding and language models. It processes a collection of articles, generates embeddings for text chunks, and utilizes cosine similarity to find the most relevant information to answer user queries accurately.

## Features

- **Environment Configuration**: Loads API keys securely using environment variables.
- **Data Handling**: Reads articles from a CSV file and splits them into manageable text chunks.
- **Embedding Generation**: Creates numerical representations of text chunks using OpenAI's embedding API.
- **Similarity Computation**: Calculates cosine similarity between user queries and document embeddings to retrieve relevant chunks.
- **Answer Generation**: Constructs prompts with retrieved context and generates answers using OpenAI's language models.
- **Robust Error Handling**: Incorporates error management to handle potential issues during embedding and completion processes.

## Dependencies

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
