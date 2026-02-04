# Sentiment Analysis using Hugging Face Pretrained Model

from transformers import pipeline

# Load pretrained sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Dataset (same as Task 1)
texts = [
    "I love this product",
    "Very bad experience",
    "Amazing quality",
    "Not worth the money",
    "Disappointed"
]

# Analyze sentiment
print("Sentiment Analysis Results:\n")
for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"Prediction: {result['label']}, Confidence: {result['score']:.2f}\n")
