from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, GridSearchCV

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
# 4. DEFINE PARAMETERS
# =====================================

parameters = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}


# =====================================
# 5. CREATE GRID SEARCH
# =====================================

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="accuracy"
)


# =====================================
# 6. TRAIN GRID SEARCH
# =====================================

grid_search.fit(X_train, y_train)


# =====================================
# 7. BEST PARAMETERS
# =====================================

print("========== BEST PARAMETERS ==========")

print(grid_search.best_params_)


# =====================================
# 8. BEST CROSS-VALIDATION SCORE
# =====================================

print("\n========== BEST CV SCORE ==========")

print(round(grid_search.best_score_, 2))


# =====================================
# 9. TEST THE BEST MODEL
# =====================================

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n========== TEST RESULTS ==========")

print("Test accuracy:", round(accuracy, 2))