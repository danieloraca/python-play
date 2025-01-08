from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=10)
print(sum)

# response: [{'summary_text': 'Sam Shleifer writes the best'}]
print(summary[0]['summary_text'])
