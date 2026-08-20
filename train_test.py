import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data
data = {
    "Hours_Studied": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    "Exam_Score": [42, 48, 51, 58, 62, 67, 71, 76, 79, 84, 89, 95]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Hours_Studied"]]
y = df["Exam_Score"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

print("\nActual scores:")
print(y_test.values)

print("\nPredicted scores:")
print(predictions)