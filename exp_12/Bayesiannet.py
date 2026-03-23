import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Load dataset
df = pd.read_csv("C:/Users/mmaha/AppData/Local/Programs/Python/Python313/Machine learning/Laboratary/exp_12/heart.csv")

# 🔍 Select relevant columns (adjust based on your dataset)
data = df[['age', 'sex', 'cp', 'chol', 'trestbps', 'target']]

# 🔄 Convert continuous → categorical
data['age_group'] = pd.cut(data['age'], bins=3, labels=["Young","Middle","Old"])
data['chol_level'] = pd.cut(data['chol'], bins=3, labels=["Low","Medium","High"])
data['bp_level'] = pd.cut(data['trestbps'], bins=3, labels=["Low","Medium","High"])

# Keep categorical only
data = data[['target', 'sex', 'cp', 'age_group', 'chol_level', 'bp_level']]

# -------------------------------
# 🔷 Bayesian Network Structure
# -------------------------------
G = nx.DiGraph()

G.add_edges_from([
    ('target','sex'),
    ('target','cp'),
    ('target','age_group'),
    ('target','chol_level'),
    ('target','bp_level')
])

plt.figure()
nx.draw(G, with_labels=True)
plt.title("Bayesian Network Structure (Heart Dataset)")
plt.show()

# -------------------------------
# 🔷 Probability Tables
# -------------------------------

# P(target)
target_prob = data['target'].value_counts(normalize=True)

# Conditional probabilities
sex_cpt = pd.crosstab(data['sex'], data['target'], normalize='columns')
cp_cpt = pd.crosstab(data['cp'], data['target'], normalize='columns')
age_cpt = pd.crosstab(data['age_group'], data['target'], normalize='columns')
chol_cpt = pd.crosstab(data['chol_level'], data['target'], normalize='columns')
bp_cpt = pd.crosstab(data['bp_level'], data['target'], normalize='columns')

# -------------------------------
# 🔷 Graphs
# -------------------------------

# P(target)
plt.figure()
target_prob.plot(kind='bar')
plt.title("P(Heart Disease)")
plt.xlabel("Target")
plt.ylabel("Probability")
plt.show()

# Function to plot heatmap
def plot_heatmap(matrix, title):
    plt.figure()
    plt.imshow(matrix)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(matrix.columns)), matrix.columns)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.show()

plot_heatmap(sex_cpt, "P(Sex | Target)")
plot_heatmap(cp_cpt, "P(CP | Target)")
plot_heatmap(age_cpt, "P(Age | Target)")
plot_heatmap(chol_cpt, "P(Cholesterol | Target)")
plot_heatmap(bp_cpt, "P(BP | Target)")