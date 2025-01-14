import os
from dotenv import load_dotenv
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client

# Define the prompt
system_prompt = "You are an assistant finding information about TV series."
human_prompt = "What is the plot of the series The Big bang theory?"

full_prompt = f"{system_prompt}\n\n{human_prompt}"

# Generate the response
response = client.chat.completions.create(
                                          model="gpt-3.5-turbo",
                                          messages=[
                                              {"role": "system", "content": system_prompt},
                                              {"role": "user", "content": human_prompt}
                                          ],
                                          temperature=0
                                      )

# Print the AI's response
print(response.choices[0].message.content.strip())
