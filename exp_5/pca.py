import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load sklearn dataset (numerical regression dataset)
data = load_diabetes()

X = data.data        # Features
y = data.target     # Target values (continuous)

# Create pass/fail labels (classification)
# Using median value as threshold
threshold = np.median(y)
pass_fail = np.array([1 if val >= threshold else 0 for val in y])

# Standardization
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA (reduce to 1 component for simplicity)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_std)

# Split data
X_train, X_test, y_train, y_test, pf_train, pf_test = train_test_split(
    X_pca, y, pass_fail, test_size=0.2, random_state=42
)

# Train regression model
sem_reg = LinearRegression()
sem_reg.fit(X_train, y_train)

# Train classification model
clf = LogisticRegression()
clf.fit(X_train, pf_train)

# User input (same number of features as dataset)
print("\nEnter 10 feature values:")

user_input = []
for i in range(X.shape[1]):
    val = float(input(f"Feature {i+1}: "))
    user_input.append(val)

user_X = np.array([user_input])

# Transform input
user_std = scaler.transform(user_X)
user_pca = pca.transform(user_std)

# Prediction
predicted_value = sem_reg.predict(user_pca)[0]
predicted_class = clf.predict(user_pca)[0]

result = "PASS" if predicted_class == 1 else "FAIL"

print("\nPredicted Value:", round(predicted_value, 2))
print("Classification Result:", result)