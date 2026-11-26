import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

# Generate synthetic marks
np.random.seed(42)

X = np.random.randint(0, 101, size=(5000, 5))

# Generate labels
y = []

for marks in X:
    average = np.mean(marks)

    if all(mark >= 35 for mark in marks) and average >= 40:
        y.append(1)  # Pass
    else:
        y.append(0)  # Fail

# Train model
model = RandomForestClassifier(random_state=42)

model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("Student model trained and saved!")