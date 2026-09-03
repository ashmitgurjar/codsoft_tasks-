import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from data_loader import load_dataset
from preprocessing import clean_text

def train_and_evaluate_models(data_path="data/movies_dataset.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Kaggle IMDb dataset...")
    df = load_dataset(data_path)

    print(f"Preprocessing text for {len(df)} movie plots...")
    df['clean_plot'] = df['plot'].apply(clean_text)

    # Filter out empty clean plots
    df = df[df['clean_plot'].str.strip() != ""].reset_index(drop=True)

    # Encode labels
    label_encoder = LabelEncoder()
    df['genre_id'] = label_encoder.fit_transform(df['genre'])
    genre_names = list(label_encoder.classes_)

    # Train / Test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['clean_plot'], df['genre_id'], test_size=0.2, random_state=42, stratify=df['genre_id']
    )

    print(f"Dataset split: {len(X_train_raw)} training samples, {len(X_test_raw)} testing samples.")

    # TF-IDF Vectorization
    print("Vectorizing text with TF-IDF (unigrams + bigrams)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True,
        min_df=3
    )
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)

    print(f"Vocabulary size: {len(tfidf.vocabulary_)} features")

    # Define classifiers
    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=1.5, max_iter=1000, random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
    }

    metrics_summary = {}
    trained_models = {}

    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        # 5-fold cross validation for fast evaluation on subset of training data
        cv_subset_size = min(5000, len(X_train_raw))
        cv_scores = cross_val_score(model, X_train[:cv_subset_size], y_train[:cv_subset_size], cv=3, scoring='accuracy', n_jobs=-1)

        print(f"{model_name} Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"{model_name} Weighted F1-Score: {f1:.4f}")
        print(f"{model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        metrics_summary[model_name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        trained_models[model_name] = model

        # Save individual model
        model_filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(output_dir, model_filename))

    # Extract top TF-IDF keywords per genre using Logistic Regression coefficients
    top_words_per_genre = {}
    feature_names = np.array(tfidf.get_feature_names_out())
    log_reg = trained_models["Logistic Regression"]

    for idx, genre_name in enumerate(genre_names):
        top_indices = np.argsort(log_reg.coef_[idx])[-10:][::-1]
        top_words_per_genre[genre_name] = feature_names[top_indices].tolist()

    # Save artifacts
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))

    benchmark_data = {
        "models": metrics_summary,
        "genres": genre_names,
        "top_keywords": top_words_per_genre,
        "vocabulary_size": len(tfidf.vocabulary_),
        "num_train_samples": len(X_train_raw),
        "num_test_samples": len(X_test_raw)
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(benchmark_data, f, indent=4)

    print(f"\nAll models trained on Kaggle IMDb dataset successfully saved to '{output_dir}/'")
    return benchmark_data

if __name__ == "__main__":
    train_and_evaluate_models()
