# part 1: Train _ save model - run once 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import joblib

# Generate ethiopian-style diabetes data
np.random.seed(42)
n_samples = 500

data = {
    'age': np.random.normal(45, 15, n_samples).clip(20, 80),
    'bmi': np.random.normal(24, 4, n_samples).clip(15, 40),
    'glucose': np.random.normal(120, 30, n_samples).clip(70, 200),
    'blood_pressure': np.random.normal(85, 12, n_samples).clip(60, 120),
    'family_history': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'physical_activity': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
}

df = pd.DataFrame(data)

# Simulate diabetes risk score (0-100)
df['risk_score'] = (
    0.3 * (df['age'] - 70) / 130 * 100 + 
    0.2 * (df['bmi'] - 18) / 17 * 100 + 
    0.2 * (df['glucose'] - 20) / 60 * 100 +
    0.15 * df['blood_pressure'] / 120 * 100 +
    0.1 * df['family_history'] * 30 + 
    -0.05 * df['physical_activity'] * 20 + 
    np.random.normal(0, 5, n_samples)
)
df['risk_score'] = df['risk_score'].clip(0, 100)

# Features and target
features = ['age', 'bmi', 'glucose', 'blood_pressure', 'family_history', 'physical_activity']
X = df[features]
y = df['risk_score']

# Train model
model = LinearRegression()
model.fit(X, y)

# save model
joblib.dump(model, 'diabetes_model.pkl')
print("Model trained and saved as 'diabetes_model.pkl'")
# print(f"Sample prediction for age=45, bmi=27, glucose=130, bp=90, family=1, activity=1: {model.predict([[45,27,130,90,1,1]])[0]:.1f}")

# or instead of model.predict([[45,27,130,90,1,1]])
input_df = pd.DataFrame([[45,27,130,90,1,1]],
                        columns=['age', 'bmi', 'glucose', 'blood_pressure', 'family_history', 'physical_activity'])
prediction = model.predict(input_df)[0]
print(f"{prediction}")

