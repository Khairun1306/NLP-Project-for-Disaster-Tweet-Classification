import re
import joblib
import numpy as np
import scipy.sparse as sp

from pathlib import Path
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load artefacts
ARTEFACTS = Path("artefacts")

best_model = joblib.load(ARTEFACTS / "best_model.pkl")
tfidf = joblib.load(ARTEFACTS / "tfidf_vectorizer.pkl")
scaler = joblib.load(ARTEFACTS / "scaler.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(tokens)


def extract_meta(text):
    cleaned = clean_text(text)

    meta = np.array([[
        int("#" in text),
        int("@" in text),
        int("http" in text.lower() or "www" in text.lower()),
        len(text),
        len(cleaned.split()),
        text.count("!"),
        text.count("?"),
        sum(c.isupper() for c in text) / max(len(text), 1),
        TextBlob(text).sentiment.polarity,
        TextBlob(text).sentiment.subjectivity
    ]], dtype=float)

    return sp.csr_matrix(meta)


def predict_tweet(tweet):

    if not tweet.strip():
        return " Please enter a tweet.", ""

    cleaned = clean_text(tweet)

    tfidf_vec = tfidf.transform([cleaned])
    meta_vec = extract_meta(tweet)

    X = sp.hstack([tfidf_vec, meta_vec])
    X = scaler.transform(X)

    prediction = best_model.predict(X)[0]

    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(X)[0]
        confidence = f"Disaster: {prob[1]:.1%}\nNon-Disaster: {prob[0]:.1%}"
    else:
        confidence = "Probability not available."

    if prediction == 1:
        result = " Disaster Tweet"
    else:
        result = " Non-Disaster Tweet"

    return result, confidence