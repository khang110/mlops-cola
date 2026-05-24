from fastapi import FastAPI
from mangum import Mangum
from inference_onnx import ColaONNXPredictor
app = FastAPI(title="MLOps Basics App")

@app.get("/")
async def home():
    return "<h2>This is a sample NLP Project</h2>"

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = ColaONNXPredictor("./models/model.onnx")
    return predictor

@app.get("/predict")
async def get_prediction(text: str):
    pred = get_predictor()
    result = pred.predict(text)
    return result


def handler(event, context):
    # Check if this is an HTTP/API Gateway/Function URL event or a direct invocation
    if "httpMethod" in event or "requestContext" in event or "rawPath" in event:
        # It's an HTTP/Mangum event
        asgi_handler = Mangum(app)
        return asgi_handler(event, context)
    
    # It's a direct invocation test event!
    text = event.get("sentence") or event.get("text")
    if not text:
        return {
            "statusCode": 400,
            "body": "Missing 'sentence' or 'text' key in event payload"
        }
    
    pred = get_predictor()
    predictions = pred.predict(text)
    return {
        "statusCode": 200,
        "body": predictions
    }