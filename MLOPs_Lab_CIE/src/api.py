from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

class InputData(BaseModel):
    course_hours: int = Field(..., ge=10, le=200)
    quizzes_count: int = Field(..., ge=5, le=50)
    difficulty_level: int = Field(..., ge=1, le=5)
    learner_experience: int = Field(..., ge=1, le=5)

@app.get("/ping")
def ping():
    return {"alive": True, "service": "EduTrack completion_days API"}

@app.post("/forecast")
def forecast(data: InputData):
    features = np.array([[ 
        data.course_hours,
        data.quizzes_count,
        data.difficulty_level,
        data.learner_experience
    ]])

    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}