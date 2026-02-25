# Sentiment Analysis using Naïve Bayes (From Scratch)
# Dataset imported from sklearn

from sklearn.datasets import fetch_20newsgroups

# Load dataset (selecting two categories for binary classification)
categories = ['rec.sport.baseball', 'sci.med']
data = fetch_20newsgroups(subset='train', categories=categories)

# Convert into (text, label) format
dataset = []

for text, target in zip(data.data, data.target):
    if target == 0:
        dataset.append((text.lower(), "Positive"))
    else:
        dataset.append((text.lower(), "Negative"))

# Split documents by class
positive_docs = []
negative_docs = []

for text, label in dataset:
    if label == "Positive":
        positive_docs.append(text)
    else:
        negative_docs.append(text)

# Count documents
total_docs = len(dataset)
positive_count = len(positive_docs)
negative_count = len(negative_docs)

# Prior probabilities
P_positive = positive_count / total_docs
P_negative = negative_count / total_docs

# Function to calculate word frequencies
def word_frequency(documents):
    freq = {}
    for doc in documents:
        for word in doc.split():
            freq[word] = freq.get(word, 0) + 1
    return freq

# Word frequencies
positive_freq = word_frequency(positive_docs)
negative_freq = word_frequency(negative_docs)

# Vocabulary
vocabulary = set(positive_freq.keys()).union(set(negative_freq.keys()))
vocab_size = len(vocabulary)

# Total words in each class
positive_words = sum(positive_freq.values())
negative_words = sum(negative_freq.values())

# Function to classify new sentence
def classify_sentence(sentence):
    sentence = sentence.lower()
    words = sentence.split()

    positive_prob = P_positive
    negative_prob = P_negative

    for word in words:
        positive_word_prob = (positive_freq.get(word, 0) + 1) / (positive_words + vocab_size)
        negative_word_prob = (negative_freq.get(word, 0) + 1) / (negative_words + vocab_size)

        positive_prob *= positive_word_prob
        negative_prob *= negative_word_prob

    if positive_prob > negative_prob:
        return "Positive"
    else:
        return "Negative"

# Test input
test_sentence = input("\nEnter a sentence to classify its sentiment: ")
result = classify_sentence(test_sentence)

print("\nTest Sentence:", test_sentence)
print("Predicted Sentiment:", result)