import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ==============================
# 1. CREATE DATA
# ==============================

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95],
    "Sleep": [5, 6, 6, 7, 7, 8, 8, 9],
    "Previous_Score": [45, 50, 55, 60, 68, 75, 82, 88],
    "Exam_Score": [42, 48, 58, 67, 76, 84, 89, 95]
}

df = pd.DataFrame(data)


# ==============================
# 2. FEATURES AND TARGET
# ==============================

X = df[
    [
        "Hours_Studied",
        "Attendance",
        "Sleep",
        "Previous_Score"
    ]
]

y = df["Exam_Score"]


# ==============================
# 3. SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ==============================
# 4. CREATE MODEL
# ==============================

model = LinearRegression()


# ==============================
# 5. TRAIN MODEL
# ==============================

model.fit(X_train, y_train)


# ==============================
# 6. MAKE PREDICTIONS
# ==============================

predictions = model.predict(X_test)


print("========== PREDICTIONS ==========")

for actual, predicted in zip(y_test, predictions):
    print(
        "Actual:",
        actual,
        "| Predicted:",
        round(predicted, 2)
    )


# ==============================
# 7. EVALUATE MODEL
# ==============================

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# ==============================
# 8. SEE WHAT THE MODEL LEARNED
# ==============================

print("\n========== MODEL COEFFICIENTS ==========")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 3))

print("\nIntercept:", round(model.intercept_, 3))

new_student = pd.DataFrame({
    "Hours_Studied": [6.5],
    "Attendance": [88],
    "Sleep": [7],
    "Previous_Score": [78]
})

prediction = model.predict(new_student)

print("\n========== NEW STUDENT ==========")
print("Predicted exam score:", round(prediction[0], 2))