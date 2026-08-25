from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# =====================================
# 1. LOAD DATA
# =====================================

iris = load_iris()

X = iris.data
y = iris.target


# =====================================
# 2. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# 3. CREATE PIPELINE
# =====================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])


# =====================================
# 4. TRAIN
# =====================================

pipeline.fit(X_train, y_train)


# =====================================
# 5. MAKE PREDICTIONS
# =====================================

predictions = pipeline.predict(X_test)


# =====================================
# 6. EVALUATE
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("========== MODEL RESULTS ==========")

print("Accuracy:", round(accuracy, 2))