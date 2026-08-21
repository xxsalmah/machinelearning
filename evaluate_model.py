import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ==============================
# 1. CREATE THE DATA
# ==============================

data = {
    "Hours_Studied": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    "Exam_Score": [42, 48, 51, 58, 62, 67, 71, 76, 79, 84, 89, 95]
}

df = pd.DataFrame(data)


# ==============================
# 2. SEPARATE FEATURES AND TARGET
# ==============================

X = df[["Hours_Studied"]]
y = df["Exam_Score"]


# ==============================
# 3. SPLIT THE DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=48
)


# ==============================
# 4. CREATE AND TRAIN MODEL
# ==============================

model = LinearRegression()

model.fit(X_train, y_train)


# ==============================
# 5. MAKE PREDICTIONS
# ==============================

predictions = model.predict(X_test)


# ==============================
# 6. DISPLAY ACTUAL VS PREDICTED
# ==============================

print("========== RESULTS ==========")

for actual, predicted in zip(y_test, predictions):
    print(
        "Actual:",
        actual,
        "| Predicted:",
        round(predicted, 2)
    )


# ==============================
# 7. MEAN ABSOLUTE ERROR
# ==============================

mae = mean_absolute_error(y_test, predictions)

print("\n========== MODEL ERROR ==========")
print("Mean Absolute Error:", round(mae, 2))


# ==============================
# 8. R2 SCORE
# ==============================

r2 = r2_score(y_test, predictions)

print("\n========== MODEL SCORE ==========")
print("R2 Score:", round(r2, 2))