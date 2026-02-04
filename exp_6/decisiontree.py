import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# ----------------------------------
# STEP 1: DATASET
# ----------------------------------
data = {
    'Study_Hours': ['High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High', 'Low'],
    'Attendance': ['Good', 'Average', 'Poor', 'Good', 'Poor', 'Average', 'Good', 'Poor'],
    'Previous_Result': ['Pass', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail'],
    'Extra_Classes': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Result': ['Pass', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail']
}

df = pd.DataFrame(data)

# ----------------------------------
# STEP 2: LABEL ENCODING
# ----------------------------------
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# ----------------------------------
# STEP 3: FEATURE & TARGET SPLIT
# ----------------------------------
X = df.drop('Result', axis=1)
y = df['Result']

# ----------------------------------
# STEP 4: DECISION TREE MODEL
# ----------------------------------
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# ----------------------------------
# STEP 5: PREDICTION
# ----------------------------------
# Sample: High Study, Good Attendance, Pass, Yes
new_sample = pd.DataFrame([[2, 2, 1, 1]], columns=X.columns)
prediction = model.predict(new_sample)

print("Predicted Student Result:", "PASS" if prediction[0] == 1 else "FAIL")

# ----------------------------------
# STEP 6: TREE VISUALIZATION
# ----------------------------------
plt.figure(figsize=(12, 8))
tree.plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=['Fail', 'Pass'],
    filled=True
)
plt.show()
