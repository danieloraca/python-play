from transformers import pipeline

classifier = pipeline("text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment")

result = classifier("""this is pretty interestiong""")

print(result)

# response: [{'label': '4 stars', 'score': 0.485343873500824}]
