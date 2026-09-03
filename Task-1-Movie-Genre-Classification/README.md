# Movie Genre Classification ML Engine (Kaggle IMDb Dataset)

An end-to-end Machine Learning pipeline and interactive web application that predicts movie genres from plot summaries using **TF-IDF Vectorization** (unigrams + bigrams) trained on the official **[Kaggle IMDb Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)** across **26 movie genres**.

---

## 🌟 Key Features

- **Dataset**: Official Kaggle dataset `hijest/genre-classification-dataset-imdb` containing **29,029 real IMDb plot summaries**.
- **Multi-Model Machine Learning Engine**:
  - **Multinomial Naive Bayes (MNB)**: Fast probabilistic baseline.
  - **Logistic Regression (LR)**: L2 regularized linear model.
  - **Support Vector Machine (LinearSVC with Calibrated Probabilities)**: Maximum-margin classifier.
  - **Random Forest (RF)**: Ensemble decision tree benchmark.
  - **Multi-Model Ensemble Vote**: Soft probability averaging across models.
- **Natural Language Preprocessing**:
  - HTML & noise cleaning, lowercasing, non-alphabetic filtering.
  - Stopword removal and Porter Stemming / WordNet Lemmatization via NLTK.
  - TF-IDF Vectorization with sublinear scaling and unigram/bigram feature extraction (12,000 vocabulary terms).
- **Model Explainability**:
  - Extracts key genre-driving TF-IDF terms for each input plot.
- **Interactive Dark-Mode Visual Dashboard**:
  - Real-time plot input classifier with live probability distribution bar charts.
  - 1-Click preset sample movie plots (*Interstellar*, *Superbad*, *The Dark Knight*, *Knives Out*, etc.).
  - Side-by-side performance benchmarking table for 26 genres.

---

## 📁 Repository Structure

```text
ml_internship task/
├── data/
│   └── movies_dataset.csv          # Processed 29,029 Kaggle IMDb dataset
├── models/
│   ├── tfidf_vectorizer.joblib     # Serialized TF-IDF vectorizer (12k terms)
│   ├── label_encoder.joblib        # Label encoder mapping (26 genres)
│   ├── naive_bayes.joblib          # Trained Naive Bayes model
│   ├── logistic_regression.joblib  # Trained Logistic Regression model
│   ├── support_vector_machine.joblib # Trained SVM model
│   ├── random_forest.joblib        # Trained Random Forest model
│   └── metrics.json                # Benchmark metrics & top keywords
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Kaggle dataset downloader & parser
│   ├── preprocessing.py            # Text cleaning & NLTK lemmatizer
│   ├── model_trainer.py            # ML pipeline & cross-validation
│   └── predict.py                  # Predictor class & explainability
├── static/
│   ├── style.css                   # Glassmorphism dark mode UI stylesheet
│   └── script.js                   # Async API integration & dynamic DOM
├── templates/
│   └── index.html                  # Main web dashboard interface
├── tests/
│   └── test_pipeline.py            # Automated unit testing suite
├── app.py                          # Flask web server & REST API
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## ⚡ Quick Start & Run Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Kaggle Dataset & Train Models
```bash
python3 src/model_trainer.py
```

### 3. Run Automated Tests
```bash
python3 -m unittest discover -s tests
```

### 4. Launch Web Application
```bash
python3 app.py
```
Open your browser and navigate to `http://127.0.0.1:5005` to interact with the web dashboard.

---

## 📊 Benchmark Results (26 Genres)

| Model | Test Accuracy | Weighted F1-Score | Weighted Precision | Weighted Recall |
|---|---|---|---|---|
| **Logistic Regression** | **50.10%** | **0.4895** | **0.4880** | **0.5010** |
| **Support Vector Machine** | **47.90%** | **0.4665** | **0.4674** | **0.4790** |
| **Multinomial Naive Bayes** | **47.45%** | **0.4552** | **0.4566** | **0.4745** |
| **Random Forest** | **37.41%** | **0.3387** | **0.3702** | **0.3741** |
