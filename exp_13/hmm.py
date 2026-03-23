import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("./hmm.csv")

print(df.head())

states = list(df['State'].unique())
observations = list(df['Observation'].unique())

state_map = {s: i for i, s in enumerate(states)}
obs_map = {o: i for i, o in enumerate(observations)}

state_seq = [state_map[s] for s in df['State']]
obs_seq = [obs_map[o] for o in df['Observation']]

N = len(states)
M = len(observations)
T = len(state_seq)

pi = np.zeros(N)
pi[state_seq[0]] = 1  # First state

A = np.zeros((N, N))

for i in range(T - 1):
    A[state_seq[i], state_seq[i+1]] += 1

# Normalize
for i in range(N):
    if np.sum(A[i]) != 0:
        A[i] /= np.sum(A[i])

B = np.zeros((N, M))

for i in range(T):
    B[state_seq[i], obs_seq[i]] += 1

# Normalize
for i in range(N):
    if np.sum(B[i]) != 0:
        B[i] /= np.sum(B[i])

print("States:", states)
print("Observations:", observations)

print("\nInitial Probabilities (π):\n", pi)
print("\nTransition Matrix (A):\n", A)
print("\nEmission Matrix (B):\n", B)

def viterbi(pi, A, B, obs_seq):
    N = len(pi)
    T = len(obs_seq)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialization
    delta[0] = pi * B[:, obs_seq[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            prob = delta[t-1] * A[:, j]
            psi[t, j] = np.argmax(prob)
            delta[t, j] = np.max(prob) * B[j, obs_seq[t]]

    # Backtracking
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])

    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path

test_obs = ["Walk", "Shop", "Clean", "Travel"]
test_seq = [obs_map[o] for o in test_obs]

path = viterbi(pi, A, B, test_seq)

predicted_states = [states[i] for i in path]

print("\nTest Observations:", test_obs)
print("Predicted Hidden States:", predicted_states)

# 📊 1. Transition Matrix Graph
plt.figure()
plt.imshow(A)
plt.title("Transition Matrix")
plt.colorbar()

plt.xticks(range(N), states)
plt.yticks(range(N), states)

plt.xlabel("To State")
plt.ylabel("From State")

# Add values inside cells
for i in range(N):
    for j in range(N):
        plt.text(j, i, f"{A[i,j]:.2f}", ha='center', va='center')

plt.show()


# 📊 2. Emission Matrix Graph
plt.figure()
plt.imshow(B)
plt.title("Emission Matrix")
plt.colorbar()

plt.xticks(range(M), observations)
plt.yticks(range(N), states)

plt.xlabel("Observations")
plt.ylabel("States")

# Add values inside cells
for i in range(N):
    for j in range(M):
        plt.text(j, i, f"{B[i,j]:.2f}", ha='center', va='center')

plt.show()


# 📊 3. State Distribution Graph
state_counts = df['State'].value_counts()

plt.figure()
state_counts.plot(kind='bar')

plt.title("State Distribution")
plt.xlabel("States")
plt.ylabel("Frequency")

# Show values on bars
for i, val in enumerate(state_counts):
    plt.text(i, val, str(val), ha='center', va='bottom')

plt.show()