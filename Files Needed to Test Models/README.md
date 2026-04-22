# Reddit Classifier

## Overview
This script takes a Reddit post as input and predicts its category using three different models:
- Transformer (DistilBERT)
- Naive Bayes
- Logistic Regression

For each model, it outputs:
- The predicted label  
- The top 5 most likely categories  

---

## Required Files
Place all of the following files in the same folder as the script:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `labels.pkl`
- `nb_model.pkl`
- `lr_model.pkl`

---

## Installation
Install the required packages:

```bash
pip install torch transformers joblib numpy
```

---

## Running the Script

1. Open a terminal in the project folder  
2. Run:

```bash
python classifier.py
```

3. Paste a Reddit post (supports multiple lines)  
4. Press Enter on a blank line to submit  
5. Type `q` and press Enter to exit  

---

## Functionality
- Loads the trained transformer model and tokenizer  
- Loads saved Naive Bayes and Logistic Regression models  
- Accepts multi-line user input  
- Generates predictions from all three models  
- Displays top 5 predicted categories for each model  

---

## Notes
- Uses GPU automatically if CUDA is available; otherwise runs on CPU  
- All required files must be in the same directory as the script  
- The script will fail at startup if any required file is missing  
