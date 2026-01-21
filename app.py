from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Student Success Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://votre-frontend.vercel.app",  # À mettre à jour après
        "*"  # Temporaire pour tester
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("student_pass_model.joblib")

FEATURES = [
    "school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob",
    "reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid",
    "activities","nursery","higher","internet","romantic","famrel","freetime","goout",
    "Dalc","Walc","health","absences","G1","G2"
]

class StudentInput(BaseModel):
    data: dict

@app.get("/")
def root():
    return {"status": "ok", "message": "Student ML API is running"}

@app.post("/predict")
def predict(inp: StudentInput):
    row = {col: inp.data.get(col, None) for col in FEATURES}
    df = pd.DataFrame([row])

    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])

    return {
        "passed": pred,
        "proba_passed": round(proba, 4)
    }
