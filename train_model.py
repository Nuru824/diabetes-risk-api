# part 1: Train _ save model - run once 
import numpy as np
import pandas as pd
import xgboost as xgb
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Generate ethiopian-style diabetes data
np.random.seed(42)
n_samples = 1000

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
    # add non-linear interaction xgboost will learn this
    0.05 * df['glucose'] * df['bmi'] / 100 +
    np.random.normal(0, 3, n_samples)
)
df['risk_score'] = df['risk_score'].clip(0, 100)

# Features and target
features = ['age', 'bmi', 'glucose', 'blood_pressure', 'family_history', 'physical_activity']
X = df[features]
y = df['risk_score']

# split and train xgboost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost trained. R2 score {r2:.3f}")

# save model
joblib.dump(model, 'diabetes_model.pkl')
print("Model trained and saved as 'diabetes_model.pkl'")
# print(f"Sample prediction for age=45, bmi=27, glucose=130, bp=90, family=1, activity=1: {model.predict([[45,27,130,90,1,1]])[0]:.1f}")

# or instead of model.predict([[45,27,130,90,1,1]])
input_df = pd.DataFrame([[45,27,130,90,1,1]],
                        columns=['age', 'bmi', 'glucose', 'blood_pressure', 'family_history', 'physical_activity'])
prediction = model.predict(input_df)[0]
print(f"{prediction}")
