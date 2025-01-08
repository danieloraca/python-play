from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

# Prompt for summarization
prompt = """
    Reply with location information:
(location): {location}
"""

examples = [
    {"input": "location1",  "output": "Romania"},
    {"input": "location2",  "output": "United Kingdom"},
    {"input": "location3",  "output": "United States"}
]

location = "what's location 1 and location 2?"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. "},
        {"role": "user", "content": prompt.format(location=examples[0]["input"])},
        {"role": "assistant", "content": examples[0]["output"]},
        {"role": "user", "content": prompt.format(location=examples[1]["input"])},
        {"role": "assistant", "content": examples[1]["output"]},
        {"role": "user", "content": prompt.format(location=location)}
    ]
)

print(response.choices[0].message.content)
