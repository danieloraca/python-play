from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

# Prompt for summarization
prompt = """
    Summarize the following text:
(movie): {movie}
"""

# Few-show examples
examples = [
    {"input": "Titanic",  "output": "Ship sinks"},
    {"input": "The Matrix",  "output": "Virtual reality"},
    {"input": "bamboo",  "output": "Funny forest movie"}
]

movie = "what can you tell me about the bamboo movie?"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. "},
        {"role": "user", "content": prompt.format(movie=examples[0]["input"])},
        {"role": "assistant", "content": examples[0]["output"]},
        {"role": "user", "content": prompt.format(movie=examples[1]["input"])},
        {"role": "assistant", "content": examples[1]["output"]},
        {"role": "user", "content": prompt.format(movie=examples[2]["input"])},
        {"role": "assistant", "content": examples[2]["output"]},
        {"role": "user", "content": prompt.format(movie=movie)}
    ]
)

print(response.choices[0].message.content)
