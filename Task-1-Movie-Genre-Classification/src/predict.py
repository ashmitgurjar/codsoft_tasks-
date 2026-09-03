import os
import joblib
import numpy as np
from src.preprocessing import clean_text

class GenrePredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        self.label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
        self.genres = list(self.label_encoder.classes_)

        self.models = {
            "Naive Bayes": joblib.load(os.path.join(models_dir, "naive_bayes.joblib")),
            "Logistic Regression": joblib.load(os.path.join(models_dir, "logistic_regression.joblib")),
            "Support Vector Machine": joblib.load(os.path.join(models_dir, "support_vector_machine.joblib")),
            "Random Forest": joblib.load(os.path.join(models_dir, "random_forest.joblib"))
        }

    def predict(self, text, model_name="Logistic Regression"):
        if model_name not in self.models and model_name != "Ensemble":
            model_name = "Logistic Regression"

        cleaned = clean_text(text)
        if not cleaned.strip():
            return {
                "error": "Input text contains no valid words after preprocessing."
            }

        vec = self.tfidf.transform([cleaned])

        if model_name == "Ensemble":
            # Average probabilities across models
            all_probas = []
            for name, m in self.models.items():
                if hasattr(m, "predict_proba"):
                    all_probas.append(m.predict_proba(vec)[0])
            probabilities = np.mean(all_probas, axis=0)
        else:
            model = self.models[model_name]
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vec)[0]
            else:
                decision = model.decision_function(vec)[0]
                # Softmax normalization for decision scores
                exp_scores = np.exp(decision - np.max(decision))
                probabilities = exp_scores / exp_scores.sum()

        top_indices = np.argsort(probabilities)[::-1]

        predictions = []
        for idx in top_indices:
            genre = self.genres[idx]
            prob = float(probabilities[idx])
            predictions.append({
                "genre": genre,
                "probability": round(prob, 4),
                "percentage": round(prob * 100, 2)
            })

        # Feature explainability: Find input terms with high TF-IDF values
        feature_names = np.array(self.tfidf.get_feature_names_out())
        input_vector_dense = vec.toarray()[0]
        active_indices = np.where(input_vector_dense > 0)[0]

        explainable_words = []
        if len(active_indices) > 0:
            sorted_active = active_indices[np.argsort(input_vector_dense[active_indices])[::-1]]
            for idx in sorted_active[:8]:
                explainable_words.append({
                    "word": feature_names[idx],
                    "tfidf_score": round(float(input_vector_dense[idx]), 4)
                })

        return {
            "top_genre": predictions[0]["genre"],
            "top_confidence": predictions[0]["percentage"],
            "model_used": model_name,
            "predictions": predictions,
            "key_features": explainable_words,
            "cleaned_text": cleaned
        }

if __name__ == "__main__":
    predictor = GenrePredictor()
    test_plot = "A brave astronaut crew embarks on a mission through a mysterious wormhole near Saturn to find a habitable planet for humanity."
    result = predictor.predict(test_plot, model_name="Support Vector Machine")
    print(f"Top Genre: {result['top_genre']} ({result['top_confidence']}%)")
    print("Top 3 Predictions:", result['predictions'][:3])
