import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# load the trained model once when the server starts
bundle = joblib.load("model/model.joblib")
model = bundle["model"]
MODEL_VERSION = bundle["version"]

app = FastAPI()


# all 10 features are required; missing or non-numeric -> 422 error
class Patient(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(patient: Patient):
    X = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(X)[0]
    return {"prediction": float(prediction)}
