from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import MaternalHealthRiskModel

app = FastAPI()

origins = [
    "https://maternal-health-risk.vercel.app/", 
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

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/")
def get_result(data: dict):
    print(data)
    try:
        prediction = model.predict(data)  
        return prediction  
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
