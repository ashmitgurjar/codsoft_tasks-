import os
import io
import json
import re
import pandas as pd
import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_artifacts():
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    models_path = os.path.join(MODEL_DIR, "models.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    
    if not (os.path.exists(vectorizer_path) and os.path.exists(models_path)):
        from train_model import train_and_evaluate
        train_and_evaluate()
        
    vectorizer = joblib.load(vectorizer_path)
    models = joblib.load(models_path)
    
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        
    return vectorizer, models, metrics_data

vectorizer, models, metrics_data = load_artifacts()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' url_link ', text)
    text = re.sub(r'\d+', ' number_token ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_keywords(text, vectorizer, nb_model):
    """XAI: Analyzes individual words in text to identify spam-indicative words."""
    clean = preprocess_text(text)
    words = text.split()
    feature_names = vectorizer.get_feature_names_out()
    feature_map = {feat: idx for idx, feat in enumerate(feature_names)}
    
    spam_prob_log = nb_model.feature_log_prob_[1]
    ham_prob_log = nb_model.feature_log_prob_[0]
    spam_ratio = spam_prob_log - ham_prob_log
    
    word_analysis = []
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        score = 0.0
        is_spam_word = False
        if clean_word in feature_map:
            idx = feature_map[clean_word]
            score = float(spam_ratio[idx])
            if score > 0.5:
                is_spam_word = True
        word_analysis.append({
            "token": word,
            "score": round(score, 3),
            "is_spam_indicator": is_spam_word
        })
    return word_analysis

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics_data)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    model_name = data.get("model_name", "Naive Bayes")
    
    if not text:
        return jsonify({"error": "No SMS text provided"}), 400
        
    if model_name not in models:
        model_name = "Naive Bayes"
        
    clean_t = preprocess_text(text)
    vec = vectorizer.transform([clean_t])
    
    selected_clf = models[model_name]
    is_spam_pred = int(selected_clf.predict(vec)[0])
    
    if hasattr(selected_clf, "predict_proba"):
        proba = selected_clf.predict_proba(vec)[0]
        spam_prob = float(proba[1])
        ham_prob = float(proba[0])
    else:
        spam_prob = 1.0 if is_spam_pred == 1 else 0.0
        ham_prob = 1.0 - spam_prob
        
    all_predictions = {}
    for name, clf in models.items():
        p = int(clf.predict(vec)[0])
        prob = float(clf.predict_proba(vec)[0][1]) if hasattr(clf, "predict_proba") else (1.0 if p == 1 else 0.0)
        all_predictions[name] = {
            "is_spam": p == 1,
            "label": "SPAM" if p == 1 else "LEGITIMATE (HAM)",
            "spam_probability": round(prob * 100, 2)
        }
        
    word_keywords = analyze_keywords(text, vectorizer, models["Naive Bayes"])
    
    return jsonify({
        "input_text": text,
        "selected_model": model_name,
        "is_spam": is_spam_pred == 1,
        "label": "SPAM" if is_spam_pred == 1 else "LEGITIMATE (HAM)",
        "spam_probability": round(spam_prob * 100, 2),
        "ham_probability": round(ham_prob * 100, 2),
        "confidence_score": round(max(spam_prob, ham_prob) * 100, 2),
        "all_models": all_predictions,
        "word_analysis": word_keywords
    })

@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    uploaded_file = request.files["file"]
    filename = uploaded_file.filename.lower()
    
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            # Find message column
            possible_cols = [c for c in df.columns if any(k in c.lower() for k in ["text", "sms", "message", "msg", "v2", "content"])]
            text_col = possible_cols[0] if possible_cols else df.columns[0]
            texts = df[text_col].astype(str).tolist()
        elif filename.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            texts = [line.strip() for line in content.split("\n") if line.strip()]
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV or TXT file."}), 400
            
        results = []
        spam_cnt = 0
        ham_cnt = 0
        
        clf = models["Naive Bayes"]
        
        for text in texts[:200]: # process up to 200 items in batch
            clean = preprocess_text(text)
            vec = vectorizer.transform([clean])
            pred = int(clf.predict(vec)[0])
            prob = float(clf.predict_proba(vec)[0][1])
            
            is_s = pred == 1
            if is_s:
                spam_cnt += 1
            else:
                ham_cnt += 1
                
            results.append({
                "text": text,
                "label": "SPAM" if is_s else "HAM",
                "is_spam": is_s,
                "spam_probability": round(prob * 100, 1)
            })
            
        return jsonify({
            "total_processed": len(results),
            "spam_count": spam_cnt,
            "ham_count": ham_cnt,
            "spam_percentage": round((spam_cnt / max(1, len(results))) * 100, 1),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route("/api/retrain", methods=["POST"])
def retrain():
    global vectorizer, models, metrics_data
    try:
        from train_model import train_and_evaluate
        metrics_data = train_and_evaluate()
        vectorizer, models, metrics_data = load_artifacts()
        return jsonify({"message": "Model retrained successfully!", "metrics": metrics_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Spam SMS Classifier Web Server on http://127.0.0.1:5050 ...")
    app.run(host="0.0.0.0", port=5050, debug=True)
