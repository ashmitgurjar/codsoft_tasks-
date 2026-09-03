import os
import json
import unittest
from src.data_loader import load_dataset
from src.preprocessing import clean_text
from src.predict import GenrePredictor
from app import app

class TestMovieGenrePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        cls.predictor = GenrePredictor(models_dir=models_dir)
        cls.client = app.test_client()

    def test_01_preprocessing(self):
        sample_text = "A team of SPECIAL forces operatives rescue kidnapped hostages in Tokyo! 100% Action-packed!"
        cleaned = clean_text(sample_text)
        self.assertNotIn("100", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertIn("special", cleaned.lower())
        self.assertIn("rescue", cleaned.lower())

    def test_02_data_loader(self):
        df = load_dataset()
        self.assertGreater(len(df), 100)
        self.assertIn("plot", df.columns)
        self.assertIn("genre", df.columns)

    def test_03_prediction_naive_bayes(self):
        text = "A haunted Victorian mansion with ghosts, supernatural hauntings, demonic possessions and terrifying murders."
        res = self.predictor.predict(text, model_name="Naive Bayes")
        self.assertIn("top_genre", res)
        self.assertEqual(res["top_genre"], "Horror")
        self.assertGreater(res["top_confidence"], 0)

    def test_04_prediction_logistic_regression(self):
        text = "An astronaut crew embarks on a deep space mission through a wormhole to explore distant galaxies."
        res = self.predictor.predict(text, model_name="Logistic Regression")
        self.assertEqual(res["top_genre"], "Sci-Fi")

    def test_05_prediction_svm(self):
        text = "Two goofy high school friends scheme a hilarious plan to buy booze for a wild house party with funny cops."
        res = self.predictor.predict(text, model_name="Support Vector Machine")
        self.assertEqual(res["top_genre"], "Comedy")

    def test_06_prediction_ensemble(self):
        text = "A passionate floral designer falls in love with a romantic photographer during a summer vacation."
        res = self.predictor.predict(text, model_name="Ensemble")
        self.assertEqual(res["top_genre"], "Romance")

    def test_07_api_predict_endpoint(self):
        payload = {
            "text": "An eccentric detective investigates a mysterious murder inside a locked room mansion with greedy suspects.",
            "model": "Logistic Regression"
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("top_genre", data)
        self.assertIn("predictions", data)

    def test_08_api_metrics_endpoint(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("models", data)

    def test_09_api_samples_endpoint(self):
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreater(len(data), 0)

if __name__ == "__main__":
    unittest.main()
