import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 0 = malignant (Fail), 1 = benign (Pass)

# -----------------------------
# STEP 2: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 3: Train Decision Tree Model
# -----------------------------
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,           # limit depth for better visualization
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# STEP 4: Prediction
# -----------------------------
y_pred = model.predict(X_test)

# Take one sample prediction
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

result = "PASS (Benign)" if prediction[0] == 1 else "FAIL (Malignant)"

print("Predicted Result for First Test Sample:", result)

# -----------------------------
# STEP 5: Model Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy of the Model:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# STEP 6: Plot Decision Tree
# -----------------------------
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    feature_names=data.feature_names.tolist(),  # FIXED (convert to list)
    class_names=["Malignant", "Benign"],
    filled=True
)

plt.title("Decision Tree for Breast Cancer Dataset")
plt.show()