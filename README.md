# Reddit Text Classification Project

## Overview
This project implements a machine learning pipeline for classifying Reddit posts and comments. It includes data preprocessing, feature extraction, model training, and performance comparison across multiple approaches.

## Purpose
The goal is to demonstrate text classification on Reddit data by comparing:
- Traditional machine learning models (Naive Bayes, Logistic Regression)
- Transformer-based models (DistilBERT)

---

## Features
- CSV data loading and preprocessing  
- Text cleaning and feature extraction  
- Multi-model training:
  - Naive Bayes  
  - Logistic Regression  
  - DistilBERT (Transformer)  
- Model evaluation and comparison  
- Results visualization  
- Interactive classification script  

---

## Dataset Split & Evaluation
The dataset is split into three parts:
- Training set  
- Validation set  
- Testing set  

All reported results are computed on the **test set (15% of the total data set)**.

---

## Quick Demo (Classifier)

### Requirements
```bash
pip install torch transformers joblib numpy pandas scikit-learn matplotlib seaborn
```

### Setup
The classifier script requires all model files located in:

```
Files Needed to Test Models/
```

This folder must contain:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `labels.pkl`
- `nb_model.pkl`
- `lr_model.pkl`

### Run
```bash
python classifier.py
```

### Usage
1. Paste a Reddit post (can be multiple lines)  
2. Press Enter on a blank line  
3. View predictions from all three models  
4. Type `q` to quit  

---

## Results Summary

| Model                  | P@1   | P@3   | P@5   |
|------------------------|-------|-------|-------|
| **DistilBERT**         | **89.9%** | **97.3%** | **98.5%** |
| Logistic Regression    | 63.7% | 84.2% | 90.5% |
| Naive Bayes            | 54.7% | 77.8% | 85.8% |

---

## Full Training Pipeline

### Notebook Usage
1. Open `training_notebook.ipynb` in Jupyter or Google Colab  
2. Run cells sequentially:
   - Load Reddit dataset  
   - Preprocess text  
   - Train all three models  
   - Generate results and visualizations  
3. Export trained models to:

```
Files Needed to Test Models/
```

### Dataset
https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts

---

## Models

| Model                  | Type            | Description                              |
|------------------------|-----------------|------------------------------------------|
| **DistilBERT**         | Transformer     | Lightweight transformer for NLP tasks    |
| Logistic Regression    | Linear Model    | Efficient linear classifier              |
| Naive Bayes            | Probabilistic   | Baseline text classification model       |

---

## Requirements

```
torch
transformers
joblib
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

---

## Output
- Precision@K metrics (P@1, P@3, P@5)  
- Confusion matrices  
- Model comparison tables  
- Saved models for inference  
- Performance visualizations  

---

## Notes
- Uses GPU if CUDA is available, otherwise runs on CPU  
- Fixed random seeds for reproducibility  
- Multi-class subreddit classification  
- Classifier script requires all model files in the same directory  
