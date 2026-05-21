import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.processor = DataModule()
        # Softmax will be applied over the class dimension (dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.labels = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        # build tensors and move to model device
        input_ids = torch.tensor([processed["input_ids"]], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([processed["attention_mask"]], dtype=torch.long).to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        probs = self.softmax(logits)
        scores = probs.detach().cpu().numpy()[0].tolist()

        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": float(score)})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best-checkpoint.ckpt")
    print(predictor.predict(sentence))
