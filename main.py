from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load ML model
model = joblib.load("student_model.pkl")


# Input structure
class StudentMarks(BaseModel):
    python: int
    maths: int
    science: int
    english: int
    computer: int


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict")
def predict(data: StudentMarks):

    marks = [[
        data.python,
        data.maths,
        data.science,
        data.english,
        data.computer
    ]]

    prediction = model.predict(marks)[0]

    if prediction == 1:
        result = "PASS"
    else:
        result = "FAIL"

    return {"prediction": result}
