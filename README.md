# 🌪️ Disaster Tweet Classification — NLP Project

## Overview
End-to-end NLP project to classify tweets as **Disaster** or **Non-Disaster** using machine learning. Built across 3 structured Jupyter notebooks.

## ⭐ Project Highlights

- End-to-end NLP workflow
- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Sentiment analysis
-  Handcrafted features and sentiment analysis
- Comparison of five machine learning models
- Hyperparameter tuning using RandomizedSearchCV
- Comprehensive model evaluation
- Interactive Gradio web application for real-time tweet classification

---

## 📁 Project Structure

```
disaster-tweet-classification/
│
├── twitter_disaster.csv               ← Original dataset (7,613 tweets)
│
├── notebook1_data_exploration.ipynb   ← Part 1: Data Exploration & Preparation
├── notebook2_modelling.ipynb          ← Part 2: Feature Engineering & Model Training
├── notebook3_evaluation.ipynb         ← Part 3 & 4: Evaluation, Validation & Deployment
│
└── artefacts/                         ← Auto-created by notebooks
    ├── cleaned_tweets.csv
    ├── tfidf_vectorizer.pkl
    ├── scaler.pkl
    ├── best_model.pkl
    ├── X_train.npz / X_test.npz
    ├── y_train.npy / y_test.npy
    └── model_summary.csv
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk textblob xgboost gradio scipy joblib
```

### 2. Place the dataset
Put `twitter_disaster.csv` in the same folder as the notebooks.

### 3. Run notebooks in order
```
notebook1_data_exploration.ipynb  →  notebook2_modelling.ipynb  →  notebook3_evaluation.ipynb
```
Each notebook saves artefacts that the next notebook loads.

---

## 📓 Notebook Breakdown

### Notebook 1 — Data Exploration & Preparation
| Task | Details |
|------|---------|
| Data loading | Pandas, shape/type inspection |
| Visualisations | Class distribution, tweet length histograms, top keywords, word clouds |
| Text cleaning | Lowercase → URL removal → Punctuation strip → Tokenise → Stopword filter → Lemmatise |
| Feature engineering | TF-IDF (10k uni+bigrams) + 8 meta-features (hashtags, mentions, URLs, length, etc.) |
| Data split | 80/20 stratified train/test |
| **Saved** | `tfidf_vectorizer.pkl`, `X_train/test.npz`, `y_train/test.npy`, `cleaned_tweets.csv` |

### Notebook 2 — Feature Engineering & Model Training
| Task | Details |
|------|---------|
| Models compared | Logistic Regression, Naive Bayes, Linear SVM, Random Forest, XGBoost |
| Evaluation | 5-fold stratified cross-validation, F1 score |
| Tuning | RandomizedSearchCV (20 iterations LR / 15 XGBoost) |
| **Saved** | `best_model.pkl`, `scaler.pkl`, `model_summary.csv` |

### Notebook 3 — Evaluation, Validation & Deployment
| Task | Details |
|------|---------|
| Confusion matrices | Raw counts + normalised |
| ROC Curve | With AUC score |
| Precision-Recall Curve | With optimal threshold analysis |
| Learning curves | Bias/variance diagnosis |
| Error analysis | Sample false positives & false negatives |
| **Web App** | Gradio interface for live tweet classification |

---

## 🏆 Model Performance (expected)

| Model | CV F1 | Test F1 | ROC-AUC |
|-------|-------|---------|---------|
| Logistic Regression (tuned) | ~0.82 | ~0.81 | ~0.89 |
| XGBoost (tuned) | ~0.81 | ~0.80 | ~0.88 |
| Linear SVM | ~0.80 | ~0.79 | N/A |
| Naive Bayes | ~0.77 | ~0.76 | ~0.85 |

*Actual values will vary depending on random seed and hyperparameter search results.*

---

## 🌐 Web App
Run the last section of `notebook3_evaluation.ipynb` to launch a **Gradio** web interface where you can paste any tweet and get an instant disaster/non-disaster prediction with confidence scores.

To share publicly, change `demo.launch(share=False)` → `demo.launch(share=True)`.

---

## 📦 Dependencies
```
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud
nltk
textblob
xgboost
gradio
scipy
joblib