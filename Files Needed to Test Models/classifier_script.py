import os
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RedditClassifier:
    def __init__(self, model_path):
        model_path = os.path.abspath(model_path)

        required = [
            "config.json", "model.safetensors", "tokenizer.json", 
            "tokenizer_config.json", "labels.pkl",
            "nb_model.pkl", "lr_model.pkl"
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
        if missing:
            raise FileNotFoundError(f"Missing: {missing}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        labels_data = joblib.load(os.path.join(model_path, "labels.pkl"))
        self.id2label = {v: k for k, v in labels_data["label2id"].items()}
        
        self.nb_pipeline = joblib.load(os.path.join(model_path, "nb_model.pkl"))
        self.lr_pipeline = joblib.load(os.path.join(model_path, "lr_model.pkl"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_transformer(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            top5_ids = torch.topk(probs, k=5).indices.tolist()
        
        pred_label = self.id2label[pred_id]
        pred_conf = probs[pred_id].item()
        top5 = [(self.id2label[i], probs[i].item()) for i in top5_ids]
        return pred_label, pred_conf, top5

    def predict_nb(self, text):
        pred_label = self.nb_pipeline.predict([text])[0]
        if hasattr(self.nb_pipeline, "predict_proba"):
            probs = self.nb_pipeline.predict_proba([text])[0]
            pred_conf = float(np.max(probs))
            top5_ids = np.argsort(probs)[::-1][:5]
            top5 = [(self.nb_pipeline.classes_[i], float(probs[i])) for i in top5_ids]
        else:
            pred_conf = None
            top5 = [(pred_label, None)]
        return pred_label, pred_conf, top5

    def predict_lr(self, text):
        pred_label = self.lr_pipeline.predict([text])[0]
        if hasattr(self.lr_pipeline, "predict_proba"):
            probs = self.lr_pipeline.predict_proba([text])[0]
            pred_conf = float(np.max(probs))
            top5_ids = np.argsort(probs)[::-1][:5]
            top5 = [(self.lr_pipeline.classes_[i], float(probs[i])) for i in top5_ids]
        else:
            pred_conf = None
            top5 = [(pred_label, None)]
        return pred_label, pred_conf, top5

    def predict_all(self, text):
        t_label, t_conf, t_top5 = self.predict_transformer(text)
        nb_label, nb_conf, nb_top5 = self.predict_nb(text)
        lr_label, lr_conf, lr_top5 = self.predict_lr(text)

        print("\n" + "=" * 80)
        print("MODEL COMPARISON - TOP 5 PREDICTIONS")
        print("=" * 80)
        
        models = [
            ("Transformer", t_label, t_conf, t_top5),
            ("Naive Bayes", nb_label, nb_conf, nb_top5),
            ("Logistic Regression", lr_label, lr_conf, lr_top5)
        ]
        
        for name, label, conf, top5 in models:
            conf_str = f"{conf:.1%}" if conf is not None else "N/A"  
            print(f"{name:<15} {label:<15} {conf_str:>8}")
            print("   Top 5:", end=" ")
            for i, (lbl, prob) in enumerate(top5, 1):
                prob_str = f"{prob:.1%}" if prob is not None else "N/A"
                print(f"{i}:{lbl[:12]:<12}({prob_str:>6})", end=" ")
            print("\n")


def read_multiline():
    print("\nPaste Reddit post (blank line → classify, 'q' → quit):")
    lines = []
    while True:
        line = input()
        if line.strip().lower() == "q" and not lines:
            return None
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    model_path = r"D:\downloads\InformationRet"
    clf = RedditClassifier(model_path)

    while True:
        text = read_multiline()
        if text is None:
            break
        if text:
            clf.predict_all(text)
