# Reddit Text Classification Project

## Overview
This project implements a machine learning pipeline for classifying Reddit text data. The notebook processes Reddit posts or comments, applies various preprocessing techniques, and trains multiple classification models to categorize the text.

## Purpose
The main goal is to demonstrate text classification on social media data from Reddit, comparing different machine learning approaches including traditional methods (Naive Bayes, Logistic Regression) and modern transformer-based models (DistilBERT).

## Features
- Data loading and preprocessing from CSV files
- Text cleaning and feature extraction
- Model training with multiple algorithms:
  - Naive Bayes
  - Logistic Regression  
  - DistilBERT (transformer-based)
- Model evaluation and comparison
- Results visualization

## Requirements
- Python 3.x
- Jupyter Notebook or Google Colab
- Required libraries: pandas, numpy, scikit-learn, transformers, torch

## Usage
1. Open the notebook in Jupyter Notebook or Google Colab
2. Mount Google Drive (if using Colab)
3. Run cells sequentially to:
   - Load and preprocess data
   - Train models
   - Evaluate performance
   - View results

## Data
The notebook expects Reddit data in CSV format with text content and labels for classification.
https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts/data (Pre merge dataset)
## Models
- **Naive Bayes**: Traditional probabilistic classifier
- **Logistic Regression**: Linear model for classification
- **DistilBERT**: Lightweight transformer model for advanced NLP tasks

## Output
The notebook generates classification reports, accuracy metrics, and comparative analysis of model performance.

## Model Performance
Transformer (DistilBERT)
P@1: 89.9%
P@3: 97.3%
P@5: 98.5%

Linear Regression
P@1: 0.6372
P@3: 0.8423
P@5: 0.9047

Naive Bayes
P@1: 0.5467
P@3: 0.7779
P@5: 0.8576

