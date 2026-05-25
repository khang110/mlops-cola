"""
Lambda wrapper
"""

import json
from inference_onnx import ColaONNXPredictor

inferencing_instance = None

def get_inferencing_instance():
    global inferencing_instance
    if inferencing_instance is None:
        inferencing_instance = ColaONNXPredictor("./models/model.onnx")
    return inferencing_instance

def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of the given sentence
    """
    instance = get_inferencing_instance()
    
    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        print(f"Got the input: {body['sentence']}")
        response = instance.predict(body["sentence"])
        return {
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(response)
        }
    else:
        return instance.predict(event["sentence"])

if __name__ == "__main__":
	test = {"sentence": "this is a sample sentence"}
	lambda_handler(test, None)
