import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1: Load the dataset
# -------------------------------------------------
df = pd.read_csv("House Price Dataset.csv")

X = df[["Area", "Bedrooms"]]
y = df["Price"]

# -------------------------------------------------
# Step 2: Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Step 3: Train Random Forest model
# (Use more trees for prediction accuracy)
# -------------------------------------------------
model = RandomForestRegressor(
    n_estimators=10,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------------------------
# Step 4: USER INPUT & PREDICTION
# -------------------------------------------------
area = float(input("Enter Area (in sqft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))

new_house = pd.DataFrame([[area, bedrooms]],
                         columns=["Area", "Bedrooms"])

predicted_price = model.predict(new_house)

print("\nPredicted House Price:", round(predicted_price[0], 2))

# -------------------------------------------------
# Step 5: Model Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))

# -------------------------------------------------
# Step 6: VISUALIZE DECISION TREES
# (Train a small forest only for visualization)
# -------------------------------------------------
viz_model = RandomForestRegressor(
    n_estimators=3,
    max_depth=3,
    random_state=42
)
viz_model.fit(X_train, y_train)

# Create one figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

for i, tree in enumerate(viz_model.estimators_):
    plot_tree(
        tree,
        feature_names=["Area", "Bedrooms"],
        filled=True,
        rounded=True,
        ax=axes[i]
    )
    axes[i].set_title(f"Decision Tree {i+1}")

plt.tight_layout()
plt.show()
