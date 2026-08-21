from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
# 3. CREATE A VERY COMPLEX TREE
# =====================================

model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)


# =====================================
# 4. TRAIN
# =====================================

model.fit(X_train, y_train)


# =====================================
# 5. PREDICT TRAINING DATA
# =====================================

train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)


# =====================================
# 6. PREDICT TEST DATA
# =====================================

test_predictions = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


# =====================================
# 7. RESULTS
# =====================================

print("========== MODEL PERFORMANCE ==========")

print(
    "Training accuracy:",
    round(train_accuracy, 2)
)

print(
    "Testing accuracy:",
    round(test_accuracy, 2)
)