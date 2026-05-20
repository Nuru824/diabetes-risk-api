# Part 2: FASTAPI DEPLOYMENT (app.py)

# app.py - Deploy this to render / python anywhere

# app.py - Deploy this to Render / PythonAnywhere
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load('diabetes_model.pkl')
app = FastAPI(title="Ethiopian Diabetes Risk API")

# define input format
class PatientData(BaseModel):
    age: float
    bmi: float
    glucose: float
    blood_pressure: float
    family_history: int
    physical_activity: int
    
    @app.get("/")
    def root():
        return {"message": "Diabetes Risk Predictor API is live! POST to /predict"}
    @app.post("/predict")
    def predict(patient: PatientData):
        try:
            # convert to 2D array for sklearn
            input_array = np.array([[
                patient.age,
                patient.bmi,
                patient.glucose,
                patient.blood_pressure,
                patient.family_history,
                patient.physical_activity
            ]])
            risk = model.predict(input_array)[0]
            risk = max(0, min(100, risk)) # clamp to 0 - 100
            
            # Determine risk level
            if risk < 30:
                level = "low"
            elif risk < 60:
                level = "medium"
            else:
                level = "high"
            return {
                "risk_score": round(risk, 1),
                "risk_level": level,
                "message": f"Diabetes risk: {level} (score: {risk:.1f})"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))