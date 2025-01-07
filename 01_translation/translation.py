from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

# english text to translate
english_text = "Hello, how are you doing today?"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a snarky AI. " +
            "Translate the following English text to Scottish."},
        {"role": "user", "content": english_text},
    ]
)

# translated text
print(response.choices[0].message.content)
