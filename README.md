# TOPSIS-Based Evaluation of Pretrained Text Classification Models

This project implements a **multi-criteria decision-making (MCDM)**
framework using **TOPSIS** to evaluate and rank multiple **pretrained
transformer-based text classification models**.

Rather than selecting a model purely based on accuracy, this approach
balances **performance**, **efficiency**, and **resource constraints**,
making it suitable for real-world deployment scenarios.

------------------------------------------------------------------------

## 📌 Models Evaluated

-   **BERT-Base**
-   **DistilBERT (SST-2 fine-tuned)**
-   **Twitter-RoBERTa**
-   **Twitter-XLM-RoBERTa**
-   **RoBERTa-Large**

All models are evaluated on the **SST-2 (Stanford Sentiment Treebank
v2)** validation set using identical preprocessing.

------------------------------------------------------------------------

## 📊 Evaluation Criteria

  Criterion                         Type
  --------------------------------- -------------
  Accuracy                          Benefit (+)
  F1-score (weighted)               Benefit (+)
  Inference Time (ms/sample)        Cost (--)
  Model Size (MB)                   Cost (--)
  Number of Parameters (Millions)   Cost (--)

------------------------------------------------------------------------

## ⚖️ TOPSIS Weights

    Accuracy        : 0.30
    F1-score        : 0.25
    Inference Time  : 0.20
    Model Size      : 0.15
    Parameters      : 0.10

------------------------------------------------------------------------

## 🛠️ Methodology

1.  Load SST-2 validation samples
2.  Run inference for each model (CPU-based)
3.  Compute Accuracy and Weighted F1-score
4.  Measure average inference latency
5.  Extract model size and parameter count
6.  Construct TOPSIS decision matrix
7.  Normalize, weight, and rank alternatives

------------------------------------------------------------------------

## 📈 Results Summary

  Rank   Model             Key Insight
  ------ ----------------- --------------------------------------------
  🥇 1   DistilBERT        Best balance of performance and efficiency
  🥈 2   Twitter-RoBERTa   Good trade-off despite domain mismatch
  🥉 3   BERT-Base         Heavy model with weak fine-tuning
  4      Twitter-XLM       Multilingual overhead
  5      RoBERTa-Large     Highest accuracy but impractical cost

------------------------------------------------------------------------

## 💡 Key Takeaway

> **Higher accuracy does not necessarily imply better overall
> suitability.**\
> Lightweight models like **DistilBERT** can outperform larger models
> when efficiency and deployment constraints are considered.

------------------------------------------------------------------------

## ▶️ How to Run

``` bash
pip install transformers datasets torch scikit-learn pandas numpy tqdm sentencepiece tiktoken
python Pretrained\ Topsis.py
```

Outputs: - `models.csv` → raw evaluation metrics - `model_topsis.csv` →
TOPSIS scores and ranks

------------------------------------------------------------------------

## 📚 Notes

-   CPU-only evaluation
-   Weighted F1 used to support binary and multiclass outputs
-   All models evaluated under identical conditions

------------------------------------------------------------------------

## 📄 License

This project is intended for academic and educational use.
