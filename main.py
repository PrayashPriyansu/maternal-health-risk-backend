from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from model import MaternalHealthRiskModel

app = FastAPI()

origins = [
    "https://maternal-health-risk.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_path = "./Maternal Health Risk Data Set.csv"
model = MaternalHealthRiskModel(data_path)
model.run_analysis()


class PatientData(BaseModel):
    Age: float = Field(..., gt=10, lt=70, description="Age in years")
    SystolicBP: float = Field(..., gt=50, lt=200, description="Systolic blood pressure in mmHg")
    DiastolicBP: float = Field(..., gt=30, lt=140, description="Diastolic blood pressure in mmHg")
    BS: float = Field(..., gt=2.0, lt=20.0, description="Blood sugar in mmol/L")
    BodyTemp: float = Field(..., gt=90.0, lt=110.0, description="Body temperature in °F")
    HeartRate: float = Field(..., gt=40, lt=200, description="Heart rate in bpm")

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/")
def get_result(data: PatientData):
    try:
        prediction = model.predict(data.model_dump())
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

