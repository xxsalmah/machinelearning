import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =====================================
# 1. LOAD THE DATASET
# =====================================

iris = load_iris()


# =====================================
# 2. CREATE A DATAFRAME
# =====================================

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target


# =====================================
# 3. LOOK AT THE DATA
# =====================================

print("========== FIRST 5 ROWS ==========")
print(df.head())


print("\n========== DATA SHAPE ==========")
print(df.shape)


print("\n========== COLUMNS ==========")
print(df.columns)


# =====================================
# 4. FEATURES AND TARGET
# =====================================

X = iris.data
y = iris.target


# =====================================
# 5. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=44
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================
# 6. CREATE MODEL
# =====================================

model = LogisticRegression(
    max_iter=200
)


# =====================================
# 7. TRAIN MODEL
# =====================================

model.fit(X_train, y_train)


# =====================================
# 8. MAKE PREDICTIONS
# =====================================

predictions = model.predict(X_test)


# =====================================
# 9. EVALUATE MODEL
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n========== MODEL RESULTS ==========")

print("Accuracy:", round(accuracy, 2))


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)