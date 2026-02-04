import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

# -------------------------------------------------
# STEP 1: TRAINING DATA (KNOWN SEMESTER MARKS)
# -------------------------------------------------
# UT max = 60, AS max = 40, Semester max = 100
data = {
    "S1": {"ut1": 55, "ut2": 55, "as1": 38, "as2": 39, "sem": 82},
    "S2": {"ut1": 43, "ut2": 52, "as1": 35, "as2": 30, "sem": 70},
    "S3": {"ut1": 37, "ut2": 39, "as1": 34, "as2": 33, "sem": 56},
    "S4": {"ut1": 22, "ut2": 18, "as1": 30, "as2": 35, "sem": 45},
    "S5": {"ut1": 40, "ut2": 32, "as1": 35, "as2": 27, "sem": 60}
}

X = []
semester_marks = []
pass_fail = []

for s in data.values():
    X.append([s["ut1"], s["ut2"], s["as1"], s["as2"]])
    semester_marks.append(s["sem"])

    # Internal (40)
    internal = (s["ut1"] + s["ut2"] + s["as1"] + s["as2"]) / 5
    # Semester converted to 60
    semester_60 = (s["sem"] / 100) * 60
    final = internal + semester_60

    pass_fail.append(1 if final >= 50 else 0)

X = np.array(X)
semester_marks = np.array(semester_marks)
pass_fail = np.array(pass_fail)

# -------------------------------------------------
# STEP 2: STANDARDIZATION
# -------------------------------------------------
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# -------------------------------------------------
# STEP 3: PCA
# -------------------------------------------------
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_std)

# -------------------------------------------------
# STEP 4: TRAIN MODELS
# -------------------------------------------------
# Semester mark predictor (out of 100)
sem_reg = LinearRegression()
sem_reg.fit(X_pca, semester_marks)

# Pass/Fail classifier
clf = LogisticRegression()
clf.fit(X_pca, pass_fail)

# -------------------------------------------------
# STEP 5: USER INPUT (NO SEMESTER MARK)
# -------------------------------------------------
print("\nEnter Student Marks (UT max 60, AS max 40)")
ut1 = float(input("UT1: "))
ut2 = float(input("UT2: "))
as1 = float(input("AS1: "))
as2 = float(input("AS2: "))

user_X = np.array([[ut1, ut2, as1, as2]])
user_std = scaler.transform(user_X)
user_pca = pca.transform(user_std)

# -------------------------------------------------
# STEP 6: PREDICTION
# -------------------------------------------------
predicted_semester = sem_reg.predict(user_pca)[0]

# Internal calculation (40)
internal_user = (ut1 + ut2 + as1 + as2) / 5

# Semester converted to 60
semester_60_user = (predicted_semester / 100) * 60

# Final marks (100)
final_user = internal_user + semester_60_user

# Result
result = "PASS ✅" if final_user >= 50 else "FAIL ❌"

# -------------------------------------------------
# STEP 7: SEMESTER MARK RANGE
# -------------------------------------------------
def sem_range(mark):
    if mark < 35:
        return "0 – 35"
    elif mark < 50:
        return "35 – 50"
    elif mark < 75:
        return "50 – 75"
    elif mark < 90:
        return "75 – 90"
    else:
        return "90 – 100"

# -------------------------------------------------
# STEP 8: OUTPUT (EXACT FORMAT)
# -------------------------------------------------
predicted_sem_round = round(predicted_semester)
internal_round = round(internal_user)
semester_60_round = round(semester_60_user)
final_round = round(final_user)

print("\n--- PCA PREDICTION RESULT ---")
print(f"Predicted Semester Marks ~ {predicted_sem_round}")
print(f"Predicted Semester Mark Range: {sem_range(predicted_sem_round)}")
print(
    f"Final Marks :  "
    f"{internal_round} + "
    f"{semester_60_round} = "
    f"{final_round}"
)
print("Result:", result)
