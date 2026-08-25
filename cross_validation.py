from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# =====================================
# 1. LOAD DATA
# =====================================

iris = load_iris()

X = iris.data
y = iris.target


# =====================================
# 2. CREATE MODEL
# =====================================

model = LogisticRegression(
    max_iter=200
)


# =====================================
# 3. CROSS-VALIDATION
# =====================================

scores = cross_val_score(
    model,
    X,
    y,
    cv=2
)


# =====================================
# 4. DISPLAY RESULTS
# =====================================

print("========== CROSS-VALIDATION ==========")

print("Scores:")

for score in scores:
    print(round(score, 2))


# =====================================
# 5. AVERAGE SCORE
# =====================================

average_score = scores.mean()

print("\nAverage score:", round(average_score, 2))