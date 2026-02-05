import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


DEVICE = "cpu"
NUM_SAMPLES = 600

MODELS = {
    "BERT-Base": "bert-base-uncased",
    "DistilBERT": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "Twitter-RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "Twitter-XLM": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "RoBERTa-Large": "siebert/sentiment-roberta-large-english"
}

dataset = load_dataset("glue", "sst2")
texts = dataset["validation"]["sentence"][:NUM_SAMPLES]
labels = dataset["validation"]["label"][:NUM_SAMPLES]

results = []

for model_name, path in MODELS.items():
    print(f"\nEvaluating {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()

    preds = []
    start_time = time.time()

    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(DEVICE)

            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            preds.append(pred)

    end_time = time.time()

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    avg_latency = ((end_time - start_time) / NUM_SAMPLES) * 1000  # ms/sample

    num_params = sum(p.numel() for p in model.parameters()) / 1e6  # Millions

    model_dir = f"./tmp_{model_name}"
    model.save_pretrained(model_dir)

    size_mb = sum(
        os.path.getsize(os.path.join(root, file))
        for root, _, files in os.walk(model_dir)
        for file in files
    ) / (1024 ** 2)

    for root, _, files in os.walk(model_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    os.rmdir(model_dir)

    results.append([
        model_name,
        round(accuracy, 4),
        round(f1, 4),
        round(avg_latency, 2),
        round(size_mb, 2),
        round(num_params, 2)
    ])

columns = [
    "Model",
    "Accuracy",
    "F1-score",
    "Inference Time (ms)",
    "Model Size (MB)",
    "Parameters (M)"
]

df = pd.DataFrame(results, columns=columns)
df.to_csv("models.csv", index=False)

print("\nEvaluation complete!")
print(df)


inp = "models.csv"
w = "0.3, 0.25, 0.2, 0.15, 0.1"
imp = "+,+,-,-,-"
out = "model_topsis.csv"


try:
    df = pd.read_csv(inp)
except FileNotFoundError:
    print("Input file not found")
    sys.exit(1)

if df.shape[1] < 3:
    print("Input file must contain three or more columns")
    sys.exit(1)

data = df.iloc[:, 1:]

try:
    data = data.astype(float)
except:
    print("From 2nd to last columns must contain numeric values only")
    sys.exit(1)

weights = w.split(",")
impacts = imp.split(",")

if len(weights) != data.shape[1] or len(impacts) != data.shape[1]:
    print("Number of weights, impacts and criteria columns must be same")
    sys.exit(1)

try:
    weights = [float(i) for i in weights]
except:
    print("Weights must be numeric and comma separated")
    sys.exit(1)

for i in impacts:
    if i not in ["+", "-"]:
        print("Impacts must be either + or -")
        sys.exit(1)

norm = data.copy()
for i in range(data.shape[1]):
    d = np.sqrt(sum(data.iloc[:, i] ** 2))
    for j in range(data.shape[0]):
        norm.iat[j, i] = data.iat[j, i] / d

for i in range(norm.shape[1]):
    for j in range(norm.shape[0]):
        norm.iat[j, i] = norm.iat[j, i] * weights[i]

ideal_best = []
ideal_worst = []

for i in range(norm.shape[1]):
    if impacts[i] == "+":
        ideal_best.append(norm.iloc[:, i].max())
        ideal_worst.append(norm.iloc[:, i].min())
    else:
        ideal_best.append(norm.iloc[:, i].min())
        ideal_worst.append(norm.iloc[:, i].max())

s_plus = []
s_minus = []

for i in range(norm.shape[0]):
    s1 = 0
    s2 = 0
    for j in range(norm.shape[1]):
        s1 += (norm.iat[i, j] - ideal_best[j]) ** 2
        s2 += (norm.iat[i, j] - ideal_worst[j]) ** 2
    s_plus.append(np.sqrt(s1))
    s_minus.append(np.sqrt(s2))

score = []
for i in range(len(s_plus)):
    score.append(s_minus[i] / (s_plus[i] + s_minus[i]))

df["Topsis Score"] = score
df["Rank"] = df["Topsis Score"].rank(ascending=False, method="max").astype(int)

df.to_csv(out, index=False)
print("TOPSIS result saved to", out)
