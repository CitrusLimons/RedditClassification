# Reddit Classifier

This is a small Python script that reads a Reddit post and predicts its category using 3 different models:

- Transformer model
- Naive Bayes
- Logistic Regression

It prints the predicted label and the top 5 guesses for each model.

## Files needed

Put these files in the same folder as the script:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `labels.pkl`
- `nb_model.pkl`
- `lr_model.pkl`

## Install packages

```bash
pip install torch transformers joblib numpy
```

## How to run

1. Open a terminal in the folder with the script.
2. Run:

```bash
python your_script_name.py
```

3. Paste in a Reddit post.
4. Press Enter on a blank line to run the prediction.
5. Type `q` and press Enter to quit.

## What it does

- Loads the saved transformer model and tokenizer.
- Loads the Naive Bayes and Logistic Regression models.
- Lets you paste in multiple lines of text.
- Shows the predicted category from each model.

## Notes

- The script uses GPU if CUDA is available, otherwise it uses CPU.
- All the model files have to be in the same folder or it will fail at startup.
