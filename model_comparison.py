import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
# 3. SCALE DATA
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================
# 4. CREATE MODELS
# =====================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# =====================================
# 5. TRAIN AND TEST EACH MODEL
# =====================================

print("========== MODEL RESULTS ==========")

for name, model in models.items():

    # Tree models don't require scaling,
    # but we'll use the scaled data here
    # for consistency in this beginner comparison.

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        name,
        "→",
        round(accuracy, 2)
    )