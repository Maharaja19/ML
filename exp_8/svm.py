import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE

# -------------------------------
# 1. Load Dataset
# -------------------------------
file_path = r"C:\Users\mmaha\AppData\Local\Programs\Python\Python313\Machine learning\Laboratary\exp_8\SMSSpamCollection"

data = pd.read_csv(file_path, sep='\t', names=["label", "message"])

print("\nFirst 15 rows of dataset:\n")
print(data.head(15))

# -------------------------------
# 2. Convert Labels
# -------------------------------
data['label'] = data['label'].map({'ham':0, 'spam':1})

# -------------------------------
# 3. Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# -------------------------------
# 4. Convert Text → TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 5. Train SVM Model
# -------------------------------
model = SVC(kernel='linear')

model.fit(X_train_vec, y_train)

# -------------------------------
# 6. Prediction
# -------------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7. t-SNE Visualization
# -------------------------------
print("\nGenerating visualization...")

tsne = TSNE(n_components=2, random_state=42)

X_train_dense = X_train_vec.toarray()
X_train_2d = tsne.fit_transform(X_train_dense)

# -------------------------------
# 8. Plot Graph
# -------------------------------
plt.figure(figsize=(10,6))

# HAM messages
plt.scatter(
    X_train_2d[y_train==0,0],
    X_train_2d[y_train==0,1],
    color='red',
    label='HAM',
    alpha=0.6
)

# SPAM messages
plt.scatter(
    X_train_2d[y_train==1,0],
    X_train_2d[y_train==1,1],
    color='black',
    label='SPAM',
    alpha=0.8
)

plt.title("Spam vs Ham Visualization using t-SNE")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.show()

# -------------------------------
# 9. Test Custom Message
# -------------------------------
input_message = input("\nEnter a message to classify as SPAM or HAM: ")

msg_vec = vectorizer.transform([input_message])

prediction = model.predict(msg_vec)

if prediction[0] == 1:
    print("\nMessage is SPAM")
else:
    print("\nMessage is HAM (Not Spam)")