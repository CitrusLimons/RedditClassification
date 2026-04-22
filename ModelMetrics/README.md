# Model Evaluation and Interpretability

This module evaluates and interprets three text classification models:

- Transformer-based model
- Naive Bayes (NB)
- Logistic Regression (LR)

It provides both **quantitative performance metrics** and **feature-level interpretability**.

---

## Overview of Metrics

The script evaluates model performance using two primary approaches:

### 1. Confusion Matrix

A confusion matrix is generated for each model to visualize prediction performance across all classes.

- Rows represent **true labels**
- Columns represent **predicted labels**
- Diagonal values indicate **correct predictions**
- Off-diagonal values indicate **misclassifications**

### Saved Files

- `transformer_cm.png`
- `nb_cm.png`
- `lr_cm.png`

These plots help identify:
- Which classes are frequently confused
- Class imbalance issues
- Model-specific weaknesses

---

### 2. Classification Report

For each model, a classification report is printed containing:

- **Precision**  
  Precision = TP / (TP + FP)  
  Measures how many predicted positives are actually correct.

- **Recall**  
  Recall = TP / (TP + FN)  
  Measures how many actual positives are correctly identified.

- **F1-score**  
  F1 = 2 * (Precision * Recall) / (Precision + Recall)  
  Harmonic mean of precision and recall.

- **Support**  
  Number of true instances for each class.

### Reports Generated For

- Transformer predictions (`preds_tf`)
- Naive Bayes predictions (`preds_nb`)
- Logistic Regression predictions (`preds_lr`)

These metrics provide a detailed per-class performance breakdown.

---

## Model Interpretability

The script extracts the most important words (features) for each class from:

- Naive Bayes model
- Logistic Regression model

### Naive Bayes Feature Importance

- Uses `feature_log_prob_`
- Represents the log probability of a word given a class
- Higher values indicate stronger association with the class

For each category:
- Top 10 words with highest log-probability are displayed

---

### Logistic Regression Feature Importance

- Uses model coefficients (`coef_`)
- Positive coefficients indicate words that increase the likelihood of a class

For each category:
- Top 10 words with the largest positive coefficients are displayed

These words represent the most influential features driving classification decisions.

---

## Pipeline Structure

Both Naive Bayes and Logistic Regression models are implemented as **scikit-learn Pipelines** consisting of:

- A vectorizer (e.g., TF-IDF or CountVectorizer)
- A classifier (NB or LR)

The script dynamically extracts components using:

- `get_feature_names_out()` for vectorizers
- `feature_log_prob_` for Naive Bayes
- `coef_` for Logistic Regression

---

## Output Content
- Confusion matrix images
- Classification reports

---
## Key Takeaways

- Confusion matrices provide a **visual diagnostic** of model errors
- Classification reports provide **quantitative performance metrics**
- Feature extraction enables **model interpretability**:
  - Naive Bayes → probabilistic word importance
  - Logistic Regression → coefficient-based importance
- Comparing all three models highlights trade-offs between:
  - Accuracy
  - Interpretability
  - Generalization
