import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 3. CREATE SCALER
# =====================================

scaler = StandardScaler()


# =====================================
# 4. SCALE THE DATA
# =====================================

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =====================================
# 5. CREATE MODEL
# =====================================

model = LogisticRegression(
    max_iter=200
)


# =====================================
# 6. TRAIN MODEL
# =====================================

model.fit(X_train_scaled, y_train)


# =====================================
# 7. PREDICT
# =====================================

predictions = model.predict(X_test_scaled)


# =====================================
# 8. EVALUATE
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("========== RESULTS ==========")

print("Accuracy:", round(accuracy, 2))