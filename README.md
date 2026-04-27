# Reddit Text Classification

## Overview

This project implements a complete machine learning pipeline for classifying Reddit posts into subreddit categories. It compares traditional ML approaches (Naive Bayes, Logistic Regression) against a fine-tuned DistilBERT transformer, covering data preprocessing, feature extraction, model training, evaluation, and interactive inference.

**Dataset:** [Reddit Self-Post Classification Task (RSPCT) — Kaggle](https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts)  
**Live Demo:** [Try the classifier on Hugging Face Spaces](https://huggingface.co/spaces/CitrusLimons/Reddit_Post_Classifier)

---

## Project Structure

| File / Folder | Description |
|---|---|
| `training_notebook.ipynb` | Full Colab training pipeline — data loading, preprocessing, model training, evaluation, and export |
| `classifier_script.py` | Local interactive inference script — loads all three models and compares predictions side by side |
| `data_prep.py` | Merges `rspct.tsv` + `subreddit_info.csv`, maps subreddits to categories, and outputs `reddit_merged_categories.csv` |
| `Files Needed to Test Models/` | All saved model and tokenizer artifacts required for local inference |

---

## Quick Start

### Option A — Try the Live Demo
No setup required: [Hugging Face Spaces](https://huggingface.co/spaces/CitrusLimons/Reddit_Post_Classifier)

### Option B — Run Locally

**1. Install dependencies:**
```bash
pip install torch transformers datasets accelerate joblib numpy pandas scikit-learn matplotlib seaborn nltk jupyter
```

**2. Download model artifacts** from the Releases section or Hugging Face and place them in a single folder.

**3. Run the classifier:**
```bash
python classifier_script.py
```

**4. Use it:**
- Paste a Reddit post (multi-line supported)
- Press **Enter on a blank line** to classify
- Type `q` on an empty prompt to quit

### Option C — Train From Scratch

1. Download `rspct.tsv` and `subreddit_info.csv` from [Kaggle](https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts)
2. Run `data_prep.py` to generate `reddit_merged_categories.csv`
3. Upload `reddit_merged_categories.csv` to Google Drive at `/content/drive/MyDrive/InformationRet/`
4. Open `training_notebook.ipynb` in Google Colab and run all cells in order

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | DistilBERT training and inference |
| `transformers` | DistilBERT model and tokenizer |
| `datasets` | HuggingFace dataset loading and tokenization caching |
| `accelerate` | Mixed-precision training support |
| `scikit-learn` | TF-IDF, Naive Bayes, Logistic Regression, metrics |
| `joblib` | Model serialization |
| `nltk` | English stopwords |
| `pandas` / `numpy` | Data processing |
| `matplotlib` / `seaborn` | Confusion matrix visualization |

> GPU with CUDA is recommended for DistilBERT training and significantly speeds up inference. The pipeline auto-detects CUDA and falls back to CPU if unavailable.

---

## Data Preparation

### Source Files
- `rspct.tsv` — Reddit self-posts (title, selftext, subreddit)
- `subreddit_info.csv` — Subreddit metadata (used for category mapping)

### Preprocessing Steps
1. Join `rspct.tsv` with `subreddit_info.csv` on `subreddit`
2. Concatenate `title` + `selftext` into a single `full_text` field
3. Clean text: HTML entity decoding, URL removal, whitespace normalization
4. Drop rows with fewer than 10 characters
5. Map subreddits to broader topic categories (see table below)
6. Retain only the **top 40 categories** by post count

### Category Mapping (examples)

| Subreddits | Category |
|---|---|
| `gaming`, `leagueoflegends`, `minecraft`, `wow`, `smashbros` | Gaming & Esports |
| `technology`, `programming`, `buildapc`, `intel` | Tech & Gadgets |
| `news`, `worldnews`, `politics` | Current Affairs |
| `science`, `askscience`, `space` | Science & Space |
| `funny`, `jokes`, `dadjokes` | Comedy & Jokes |
| `relationships`, `dating_advice` | Relationships & Dating |
| `food`, `recipes`, `cooking` | Food & Cooking |
| `movies`, `television` | Movies & TV Shows |
| `fitness` | Fitness & Wellness |
| `diy`, `harley` | Hobbies |

If a `category_1` column exists in `subreddit_info.csv`, it takes priority over the manual mapping.

**Output:** `reddit_merged_categories.csv` with columns `text`, `original_subreddit`, `category`.

---

## Training Pipeline — `training_notebook.ipynb`

The notebook runs entirely in Google Colab and is split into **independent cells** — failures or reruns in later steps (e.g. evaluation) do not force you to redo earlier steps like splitting, vectorization, or tokenization.

### Dataset Split

| Split | Proportion |
|---|---|
| Train | ~74% |
| Validation | ~11% |
| Test | 15% |

Splits are stratified by `category`. Once saved, they are reused across all model training cells. Set `FORCE_RESPLIT = True` to regenerate.

### Notebook Cell Reference

| Cell | Purpose |
|---|---|
| 1 | Mount Google Drive (force remount for clean slate) |
| 2 | Define all paths and global constants |
| 3 | Verify merged CSV exists, load and inspect dataset |
| 4 | Define shared helpers: split loader, P@K metric, JSON saver |
| 5 | Optional: dataset explorer and validation stats (safe to rerun) |
| 6 | Create stratified train / val / test split |
| 7 | Load splits and print shape info for baseline setup |
| 8 | Train Naive Bayes |
| 9 | Evaluate Naive Bayes (P@K on val + test, save metrics JSON) |
| 11 | Train Logistic Regression |
| 12 | Evaluate Logistic Regression (P@K on val + test, save metrics JSON) |
| 13 | Check transformer environment (CUDA, GPU, transformers version) |
| 14 | DistilBERT label encoding prep |
| 15 | Tokenize all splits and cache to disk |
| 16 | Fine-tune DistilBERT and save best checkpoint |
| 17 | Evaluate DistilBERT (val + test accuracy and P@K) |

---

## Models

### Naive Bayes

Trained as a full sklearn Pipeline (`TfidfVectorizer` → `MultinomialNB`), so no separate vectorizer is needed at inference.

| Parameter | Value |
|---|---|
| N-gram range | (1, 2) — unigrams + bigrams |
| Max features | 10,000 |
| Min document frequency | 5 |
| Max document frequency | 0.8 |
| Stop words | NLTK English + common contractions + `lb` |
| NB smoothing (alpha) | 1.0 |
| Saved as | `nb_baseline_model.pkl` |

### Logistic Regression

Also a full sklearn Pipeline (`TfidfVectorizer` → `LogisticRegression`).

| Parameter | Value |
|---|---|
| N-gram range | (1, 2) — unigrams + bigrams |
| Max features | 8,000 |
| Min document frequency | 5 |
| Max document frequency | 0.8 |
| Stop words | Same custom list as Naive Bayes |
| Solver | `saga` (supports multiclass, parallelizable) |
| Max iterations | 500 |
| Parallelism | `n_jobs=-1` (all CPU cores) |
| Saved as | `lr_model.pkl` |

Both models are vectorized in batches of 50,000 rows to avoid memory issues with large datasets.

### DistilBERT

Fine-tuned end-to-end on this Reddit dataset (not zero-shot) using `distilbert-base-uncased` as the base model.

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max token length | 256 |
| Training epochs | 2 |
| Batch size | 16 per device |
| Warmup steps | 300 |
| Weight decay | 0.01 |
| Save & eval frequency | Every 2,000 steps |
| Best model criterion | Lowest `eval_loss` |
| Max checkpoints kept | 3 |
| Mixed precision | `fp16` if CUDA available |
| Gradient checkpointing | Enabled (reduces VRAM usage) |
| Saved as | `distilbert_results/best_model/` |

Tokenized datasets are cached to `distilbert_tokenized/` on Drive, so Cell 15 can be skipped on reruns.

---

## Results

All metrics are computed on the **held-out test set (15% of total data)**.

| Model | P@1 | P@3 | P@5 |
|---|---|---|---|
| **DistilBERT** | **89.9%** | **97.3%** | **98.5%** |
| Logistic Regression | 71.4% | 88.9% | 93.5% |
| Naive Bayes | 64.0% | 84.9% | 91.1% |

**Precision@K** measures the fraction of posts where the correct subreddit category appears in the model's top K predictions. DistilBERT outperforms Logistic Regression by ~18 percentage points at P@1 and achieves near-perfect P@5 at 98.5%.

---

## Interactive Classifier — `classifier_script.py`

Loads all three saved models from a local directory and shows top-5 predictions with confidence scores from each model side by side.

**Example output:**
```
================================================================================
MODEL COMPARISON - TOP 5 PREDICTIONS
================================================================================
Transformer     Gaming & Esports   91.3%
   Top 5: 1:Gaming & E  (91.3%)  2:Tech & Gadg  ( 3.1%)  3:Hobbies     ( 2.0%) ...

Naive Bayes     Gaming & Esports   74.8%
   Top 5: 1:Gaming & E  (74.8%)  2:Movies & TV  ( 8.2%)  3:Hobbies     ( 6.1%) ...

Logistic Reg    Gaming & Esports   68.2%
   Top 5: 1:Gaming & E  (68.2%)  2:Tech & Gadg  (12.4%)  3:Hobbies     ( 7.3%) ...
```

The script validates all required model files at startup before loading anything. Missing files are listed explicitly in the error message.

---

## Inference Files

All files must be in the **same directory** as `classifier_script.py`:

| File | Description |
|---|---|
| `config.json` | DistilBERT model architecture config |
| `model.safetensors` | Fine-tuned DistilBERT weights |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |
| `labels.pkl` | `label2id` / `id2label` mapping dict |
| `nb_baseline_model.pkl` | Saved Naive Bayes sklearn Pipeline |
| `lr_model.pkl` | Saved Logistic Regression sklearn Pipeline |

> **Note:** The model file is named `nb_baseline_model.pkl` throughout. Make sure this matches what is exported from the training notebook.

---

## All Outputs

| Output | Location | Description |
|---|---|---|
| `reddit_merged_categories.csv` | `BASE_DIR/` | Cleaned, category-mapped dataset ready for modeling |
| `train.csv`, `val.csv`, `test.csv` | `BASE_DIR/` | Stratified dataset splits |
| `nb_baseline_model.pkl` | `BASE_DIR/` | Naive Bayes sklearn Pipeline |
| `lr_model.pkl` | `BASE_DIR/` | Logistic Regression sklearn Pipeline |
| `distilbert_tokenized/` | `BASE_DIR/` | Cached HuggingFace tokenized datasets |
| `distilbert_results/best_model/` | `BASE_DIR/` | Fine-tuned DistilBERT weights + tokenizer |
| `distilbert_results/labels.pkl` | `BASE_DIR/` | `label2id` / `id2label` mapping |
| `metrics/nb_metrics.json` | `BASE_DIR/` | Naive Bayes val + test P@K scores |
| `metrics/lr_metrics.json` | `BASE_DIR/` | Logistic Regression val + test P@K scores |
| `results/transformer_cm.png` | Colab working dir | DistilBERT confusion matrix |
| `results/nb_cm.png` | Colab working dir | Naive Bayes confusion matrix |
| `results/lr_cm.png` | Colab working dir | Logistic Regression confusion matrix |

---

## Configuration Reference

| Cell | Constant | Default | Description |
|---|---|---|---|
| 2 | `RANDOM_STATE` | `42` | Seed for splits and model training |
| 2 | `TOP_K_VALUES` | `[1, 3, 5]` | K values for Precision@K evaluation |
| 6 | `FORCE_RESPLIT` | `False` | Force re-split even if files exist |
| 8 | `FORCE_RETRAIN_NB` | `True` | Set `False` to load existing NB model |
| 11 | `FORCE_RETRAIN_LR` | `True` | Set `False` to load existing LR model |
| 16 | `FORCE_RETRAIN_DISTIL` | `False` | Set `True` to re-train DistilBERT from scratch |
| 15 | `max_length` | `256` | Max token length for DistilBERT |
| 16 | `num_train_epochs` | `2` | DistilBERT training epochs |
| 16 | `per_device_train_batch_size` | `16` | DistilBERT batch size per GPU |
| 8 | `max_features` (NB) | `10,000` | TF-IDF vocabulary size for Naive Bayes |
| 11 | `max_features` (LR) | `8,000` | TF-IDF vocabulary size for Logistic Regression |
| classifier | `model_path` | `D:\downloads\InformationRet` | Directory containing all inference model files |

---

## Notes

- All notebook cells are **idempotent** — rerunning with `FORCE_*` flags set to `False` loads cached results instead of retraining
- Google Drive is force-remounted at startup (`fusermount -u`) to prevent stale mount issues between Colab sessions
- `dataloader_num_workers=0` is used during DistilBERT evaluation to prevent multiprocessing instability in Colab
- `gradient_checkpointing=True` reduces GPU memory usage at a slight speed cost during DistilBERT training
- Fixed `RANDOM_STATE = 42` is applied across all splits and model training steps for reproducibility
- Both traditional model pipelines support `predict_proba`, enabling confidence scores and P@K computation
- The DistilBERT model is fine-tuned **end-to-end** — not used zero-shot

---

## Credits

- **Transformer model:** [DistilBERT](https://huggingface.co/distilbert-base-uncased) via Hugging Face Transformers
- **Dataset:** [Reddit Self-Post Classification Task — mswarbrickjones on Kaggle](https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts)
- **Live demo:** [Hugging Face Spaces — CitrusLimons/Reddit_Post_Classifier](https://huggingface.co/spaces/CitrusLimons/Reddit_Post_Classifier)
