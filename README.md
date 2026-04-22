# Reddit Text Classification Project

## Overview
Machine learning pipeline for classifying Reddit posts/comments. Processes text data, applies preprocessing, trains multiple models, and compares performance.

## Purpose
Demonstrates text classification on Reddit social media data, comparing traditional ML (Naive Bayes, Logistic Regression) vs. modern transformers (DistilBERT).

## Features
- CSV data loading and preprocessing
- Text cleaning and feature extraction
- Multi-model training:
  - Naive Bayes
  - Logistic Regression
  - DistilBERT (transformer)
- Model evaluation and comparison
- Results visualization
- Interactive demo script

## Quick Demo
**Files needed** (in same folder as script):

**Install:**
```bash
pip install torch transformers joblib numpy pandas scikit-learn matplotlib seaborn
```

**Run:**
```bash
python demo.py
```
Paste Reddit post → Enter → See predictions from all 3 models.

## Results Summary
| Model | P@1 | P@3 | P@5 |
|-------|-----|-----|-----|
| **DistilBERT** | **89.9%** | **97.3%** | **98.5%** |
| Logistic Regression | 63.7% | 84.2% | 90.5% |
| Naive Bayes | 54.7% | 77.8% | 85.8% |

## Full Training Pipeline
**Notebook Usage:**
1. Open `training_notebook.ipynb` in Jupyter/Colab
2. Run cells sequentially:
   - Load Reddit dataset
   - Preprocess text
   - Train all 3 models
   - Generate results + visualizations
3. Save models to `Files Needed to Test Models/`

**Data:** [Reddit Selfposts Dataset](https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts)

## Models
| Model | Type | Description |
|-------|------|-------------|
| **DistilBERT** | Transformer | Lightweight transformer for advanced NLP |
| **Logistic Regression** | Linear | Linear classification model |
| **Naive Bayes** | Probabilistic | Traditional text classifier |

## Requirementstorch
transformers
joblib
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter

## Output
- Classification reports (P@1, P@3, P@5 metrics)
- Confusion matrices
- Model comparison tables
- Saved models for demo
- Performance visualizations

## Notes
- GPU supported (auto-detects CUDA)
- Reproducible results (fixed seeds)
- Multi-class Reddit subreddit prediction
- Ready for oral presentation + final report
