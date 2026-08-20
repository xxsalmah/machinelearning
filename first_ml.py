import pandas as pd
from sklearn.linear_model import LinearRegression


# Our data
data = {
    "Hours_Studied": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    "Exam_Score": [42, 48, 51, 58, 62, 67, 71, 76, 79, 84, 89, 95]
}

df = pd.DataFrame(data)


# Input (X)
X = df[["Hours_Studied"]]

# Target (y)
y = df["Exam_Score"]


# Create the model
model = LinearRegression()


# Train the model
model.fit(X, y)


# Make a prediction
hours = 4.4

prediction = model.predict([[hours]])


print("Hours studied:", hours)
print("Predicted exam score:", round(prediction[0], 2))