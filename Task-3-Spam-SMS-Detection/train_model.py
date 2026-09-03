import os
import re
import json
import ssl
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "spam.csv")

def download_uci_dataset():
    """Downloads the official UCI / Kaggle SMS Spam Collection dataset (5,572 samples)."""
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    
    print("Downloading official Kaggle UCI SMS Spam Collection dataset...")
    try:
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
        # Save clean copy locally
        df.to_csv(CSV_PATH, index=False)
        print(f"Dataset successfully downloaded and saved to {CSV_PATH}. Total samples: {len(df)}")
        return df
    except Exception as e:
        print(f"Failed to download dataset from URL: {e}")
        if os.path.exists(CSV_PATH):
            print(f"Loading existing local dataset from {CSV_PATH}...")
            return pd.read_csv(CSV_PATH)
        else:
            raise RuntimeError("Could not obtain SMS Spam dataset!")

def preprocess_text(text):
    """Clean text by lowercasing, replacing numbers & URLs, removing punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url_link ', text)  # replace URLs
    text = re.sub(r'\d+', ' number_token ', text)                # replace digits
    text = re.sub(r'[^\w\s]', ' ', text)                          # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_evaluate(force_download=False):
    if force_download or not os.path.exists(CSV_PATH):
        df = download_uci_dataset()
    else:
        df = pd.read_csv(CSV_PATH)
        # Verify if existing csv has full dataset (5,572 rows)
        if len(df) < 2000:
            df = download_uci_dataset()
        
    df['clean_text'] = df['text'].apply(preprocess_text)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['clean_text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorizer with unigrams & bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4 Core Machine Learning Classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    metrics_summary = {}
    
    print("\n--- Training Machine Learning Models ---")
    for name, clf in classifiers.items():
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        y_proba = clf.predict_proba(X_test_tfidf)[:, 1] if hasattr(clf, "predict_proba") else y_pred
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        metrics_summary[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm
        }
        trained_models[name] = clf
        print(f"[{name:22s}] Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    # Top Spam Keywords for XAI (Explainable AI)
    feature_names = np.array(vectorizer.get_feature_names_out())
    nb_model = trained_models["Naive Bayes"]
    spam_prob_log = nb_model.feature_log_prob_[1]
    ham_prob_log = nb_model.feature_log_prob_[0]
    spam_ratio = spam_prob_log - ham_prob_log
    top_spam_indices = np.argsort(spam_ratio)[::-1][:30]
    top_spam_words = [
        {"word": str(feature_names[idx]), "weight": round(float(spam_ratio[idx]), 3)}
        for idx in top_spam_indices
    ]
    
    # Save artifacts
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(trained_models, os.path.join(MODEL_DIR, "models.pkl"))
    
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "models": metrics_summary,
            "top_spam_keywords": top_spam_words,
            "total_samples": len(df),
            "spam_count": int((y == 1).sum()),
            "ham_count": int((y == 0).sum())
        }, f, indent=2)
        
    print(f"\nSaved vectorizer, models, and metrics to {MODEL_DIR}")
    return metrics_summary

if __name__ == "__main__":
    train_and_evaluate(force_download=True)
