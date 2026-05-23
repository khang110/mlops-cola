import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer

from utils import timing


class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2",
            local_files_only=True,
        )
        self.labels = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.tokenizer(
            inference_sample["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        ort_inputs = {
            "input_ids": np.expand_dims(np.array(processed["input_ids"], dtype=np.int64), axis=0),
            "attention_mask": np.expand_dims(np.array(processed["attention_mask"], dtype=np.int64), axis=0),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": float(score)})
        print(predictions)
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)
