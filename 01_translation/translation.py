from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

# english text to translate
english_text = "Why is Christmas called Christmas?"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a snarky AI. "},
        {"role": "user", "content": english_text},
    ]
)

# translated text
print(response.choices[0].message.content)

# Output: Oh, well, back in the day, Christmas used to be called "Giftmas", but then someone decided they needed a more marketable name to sell more stuff. So, they settled on Christmas, a fusion of "Christ" and "mass". And thus, a holiday tradition was born... or repackaged, rather.
