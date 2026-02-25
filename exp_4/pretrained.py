# Sentiment Analysis using Hugging Face Pretrained Model with sklearn dataset

from transformers import pipeline
from sklearn.datasets import fetch_20newsgroups

def main():

    # Load pretrained sentiment analysis model
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Load text dataset from sklearn
    dataset = fetch_20newsgroups(subset='test')

    # Select few sample texts
    texts = dataset.data[:5]

    print("\n===== SENTIMENT ANALYSIS RESULTS =====\n")

    for i, text in enumerate(texts, start=1):

        # Reduce text length (model input limit safety)
        short_text = text[:300].replace("\n", " ")

        result = sentiment_analyzer(short_text)[0]

        print(f"Sample {i}")
        print("Text:", short_text)
        print("Prediction:", result['label'])
        print("Confidence:", round(result['score'], 2))
        print("-" * 60)


if __name__ == "__main__":
    main()