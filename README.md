# 🌪️ Disaster Tweet Classification using NLP

An end-to-end **Natural Language Processing (NLP)** project that classifies tweets as **Disaster** or **Non-Disaster** using Machine Learning.

The project demonstrates the complete machine learning workflow—from data exploration and preprocessing to model training, evaluation, and deployment through an interactive **Gradio** web application.

---

## 📌 Project Highlights

- End-to-end NLP pipeline
- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Sentiment analysis using TextBlob
- Feature engineering (10 handcrafted features)
- Comparison of multiple Machine Learning models
- Hyperparameter tuning using RandomizedSearchCV
- 5-Fold Stratified Cross Validation
- Comprehensive model evaluation
- Interactive Gradio web application

---

# 📂 Project Structure

```text
NLP-Project-for-Disaster-Tweet-Classification/
│
├── artefacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── cleaned_tweets.csv
│   ├── model_summary.csv
│   ├── X_train.npz
│   ├── X_test.npz
│   ├── y_train.npy
│   └── y_test.npy
│
├── images/
│   ├── target_distribution.png
│   ├── tweet_length_distribution.png
│   ├── wordcloud_disaster.png
│   ├── wordcloud_non_disaster.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── learning_curve.png
│
├── notebook1_data_exploration.ipynb
├── notebook2_modelling.ipynb
├── notebook3_evaluation.ipynb
│
├── predictor.py
├── app.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── twitter_disaster.csv
```

---

# 📊 Dataset

**Source:** Kaggle – Natural Language Processing with Disaster Tweets

**Dataset Size:** 7,613 Tweets

Target Labels:

- **1** → Disaster Tweet
- **0** → Non-Disaster Tweet

---

# 🏗️ Project Workflow

### 📘 Notebook 1 – Data Exploration & Preparation

- Load dataset
- Exploratory Data Analysis (EDA)
- Missing value analysis
- Duplicate removal
- Text preprocessing
- Sentiment analysis
- Feature engineering
- TF-IDF Vectorization
- Train/Test Split
- Save processed artefacts

---

### 🤖 Notebook 2 – Model Training

Models trained:

- Logistic Regression
- Naive Bayes
- Linear SVM
- Random Forest
- XGBoost

Includes:

- 5-Fold Stratified Cross Validation
- Hyperparameter tuning
- Model comparison
- Performance evaluation
- Confusion Matrix
- Classification Report
- Save best model

---

### 📈 Notebook 3 – Model Evaluation

Includes:

- ROC Curve
- Precision-Recall Curve
- Learning Curve
- Error Analysis
- Final Model Summary
- Deployment Overview

---

# 🧠 Feature Engineering

The final model combines **TF-IDF features** with **10 handcrafted numerical features**:

- Has hashtag
- Has mention
- Has URL
- Character count
- Word count
- Exclamation count
- Question count
- Uppercase ratio
- Sentiment polarity
- Sentiment subjectivity

---

# 🏆 Best Performing Model

**Logistic Regression**

The final model was selected based on its superior performance after hyperparameter tuning and cross-validation.

The trained model is saved as:

- `best_model.pkl`

---

# 🖥️ Application Architecture

The deployment is organized into separate modules:

| File | Purpose |
|------|---------|
| `predictor.py` | Loads artefacts, preprocesses tweets, extracts features, and performs prediction |
| `app.py` | Gradio web interface |
| `artefacts/` | Stores trained model, scaler, TF-IDF vectorizer and processed data |

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Khairun1306/NLP-Project-for-Disaster-Tweet-Classification.git
```

Move into the project directory:

```bash
cd NLP-Project-for-Disaster-Tweet-Classification
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Launch the Application

```bash
python app.py
```

The Gradio application will open automatically in your default web browser.

---

# 📷 Project Outputs

## Target Distribution

![](images/target_distribution.png)

---

## Tweet Length Distribution

![](images/tweet_length_distribution.png)

---

## Disaster Tweet Word Cloud

![](images/wordcloud_disaster.png)

---

## Non-Disaster Tweet Word Cloud

![](images/wordcloud_non_disaster.png)

---

## Confusion Matrix

![](images/confusion_matrix.png)

---

## ROC Curve

![](images/roc_curve.png)

---

## Precision-Recall Curve

![](images/precision_recall_curve.png)

---

## Learning Curve

![](images/learning_curve.png)

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TextBlob
- SciPy
- XGBoost
- Matplotlib
- WordCloud
- Gradio

---

# 👩‍💻 Author

**Khairun Nisa**

GitHub: https://github.com/Khairun1306

---

# 🙏 Acknowledgements

- Kaggle – Natural Language Processing with Disaster Tweets
- Scikit-learn Documentation
- NLTK Documentation
- Gradio Documentation