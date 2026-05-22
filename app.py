from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")

@app.get("/")
async def home():
    return "<h2>This is a sample NLP Project</h2>"



# load the model
predictor = ColaONNXPredictor("./models/model.onnx")

@app.get("/predict")
async def get_prediction(text: str):
    result =  predictor.predict(text)
    return result