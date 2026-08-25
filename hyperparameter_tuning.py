from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

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
# 3. CREATE MODEL
# =====================================

model = RandomForestClassifier(
    random_state=42
)


# =====================================
# 4. DEFINE HYPERPARAMETERS
# =====================================

parameters = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 5, None]
}


# =====================================
# 5. GRID SEARCH
# =====================================

grid_search = GridSearchCV(
    model,
    parameters,
    cv=5,
    scoring="accuracy"
)


# =====================================
# 6. TRAIN GRID SEARCH
# =====================================

grid_search.fit(X_train, y_train)


# =====================================
# 7. BEST SETTINGS
# =====================================

print("========== BEST PARAMETERS ==========")

print(grid_search.best_params_)


# =====================================
# 8. BEST CROSS-VALIDATION SCORE
# =====================================

print("\n========== BEST CV SCORE ==========")

print(
    round(
        grid_search.best_score_,
        2
    )
)


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