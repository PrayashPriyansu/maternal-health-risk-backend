from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import MaternalHealthRiskModel

app = FastAPI()

# Allow frontend (React/Vue/Next.js) to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and train model once
data_path = "./Maternal Health Risk Data Set.csv"
model = MaternalHealthRiskModel(data_path)
model.run_analysis()  # ✅ Train the model once when FastAPI starts

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/")
def get_result(data: dict):
    print(data)
    try:
        prediction = model.predict(data)  # ✅ Get predictions
        return prediction  # ✅ JSON response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
