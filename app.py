# Part 2: FASTAPI DEPLOYMENT (app.py)

# app.py - Deploy this to render / python anywhere

# app.py - Deploy this to Render / PythonAnywhere
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load('diabetes_model.pkl')

import os

# This automatically detects and adjusts to the Hugging Face proxy path
app = FastAPI(root_path=os.getenv("SPACE_HOST", ""))

@app.get("/")
def read_root():
    return {"message": "API is active!"}

# app = FastAPI(
#     title="Ethiopian Diabetes Risk API",
#     docs_url="/docs",
#     redoc_url="/redoc",
#     openapi_url="/openapi.json"
# )

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
import os

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


# app.py - Deploy this to Render / PythonAnywhere

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import os

# # Load model
# model = joblib.load("diabetes_model.pkl")

# # Create FastAPI app
# app = FastAPI(title="Ethiopian Diabetes Risk API")


# # Define input format
# class PatientData(BaseModel):
#     age: float
#     bmi: float
#     glucose: float
#     blood_pressure: float
#     family_history: int
#     physical_activity: int


# # Root endpoint
# @app.get("/")
# def root():
#     return {
#         "message": "Diabetes Risk Predictor API is live! POST to /predict"
#     }


# # Prediction endpoint
# @app.post("/predict")
# def predict(patient: PatientData):
#     try:
#         # Convert input into NumPy array
#         input_array = np.array([[
#             patient.age,
#             patient.bmi,
#             patient.glucose,
#             patient.blood_pressure,
#             patient.family_history,
#             patient.physical_activity
#         ]])

#         # Predict risk
#         risk = model.predict(input_array)[0]

#         # Clamp score between 0 and 100
#         risk = max(0, min(100, risk))

#         # Determine risk level
#         if risk < 30:
#             level = "low"
#         elif risk < 60:
#             level = "medium"
#         else:
#             level = "high"

#         return {
#             "risk_score": round(float(risk), 1),
#             "risk_level": level,
#             "message": f"Diabetes risk: {level} (score: {risk:.1f})"
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


# # Run locally
# if __name__ == "__main__":
#     import uvicorn

#     port = int(os.environ.get("PORT", 7860))

#     uvicorn.run(app, host="0.0.0.0", port=port)