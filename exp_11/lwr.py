import numpy as np
import matplotlib.pyplot as plt

# ---------------- STEP 1: GENERATE LARGE DATASET ----------------
np.random.seed(42)

n = 150  # more than 100 rows

# Features
study_hours = np.random.uniform(1, 10, n)
social_media = np.random.uniform(1, 8, n)
sleep_hours = np.random.uniform(4, 9, n)

# Target (Marks)
marks = (
    5 * study_hours 
    - 3 * social_media 
    + 2 * sleep_hours 
    + np.random.normal(0, 3, n)
)

# Add bias term (x0 = 1)
X = np.c_[np.ones(n), study_hours, social_media, sleep_hours]
y = marks


# ---------------- STEP 2: LWR FUNCTION ----------------
def lwlr(x_query, X, y, tau):
    m = X.shape[0]
    W = np.eye(m)

    for i in range(m):
        diff = x_query - X[i]
        W[i, i] = np.exp(diff @ diff.T / (-2 * tau**2))

    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return x_query @ theta


# ---------------- STEP 3: PREDICTION ----------------
tau = 1.5  # bandwidth parameter

y_pred = []
for i in range(n):
    y_pred.append(lwlr(X[i], X, y, tau))

y_pred = np.array(y_pred)


# ---------------- STEP 4: VISUALIZATION ----------------
plt.scatter(study_hours, y, label="Actual Marks")
plt.scatter(study_hours, y_pred, label="Predicted Marks")

plt.xlabel("Study Hours")
plt.ylabel("Marks (%)")
plt.title("Locally Weighted Regression (150 Samples)")
plt.legend()
plt.show()