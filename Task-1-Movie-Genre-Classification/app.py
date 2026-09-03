import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.predict import GenrePredictor

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Initialize predictor
models_dir = os.path.join(os.path.dirname(__file__), "models")
predictor = GenrePredictor(models_dir=models_dir)

def get_latest_metrics():
    metrics_file = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {}

SAMPLE_PLOTS = [
    {
        "title": "Interstellar (2014)",
        "expected_genre": "Sci-Fi",
        "plot": "In a future Earth facing agricultural crisis and dust storms, a team of astronauts travels through a wormhole near Saturn in search of a new home for humanity."
    },
    {
        "title": "The Conjuring (2013)",
        "expected_genre": "Horror",
        "plot": "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark demonic presence in their secluded farmhouse."
    },
    {
        "title": "Superbad (2007)",
        "expected_genre": "Comedy",
        "plot": "Two co-dependent high school seniors attempt to buy alcohol for a high school party, leading to a crazy night of bizarre misadventures and hilarious chaos with eccentric police officers."
    },
    {
        "title": "The Dark Knight (2008)",
        "expected_genre": "Action",
        "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
    },
    {
        "title": "La La Land (2016)",
        "expected_genre": "Romance",
        "plot": "While navigating their careers in Los Angeles, a passionate jazz pianist and an aspiring actress fall deeply in love while pursuing their Hollywood dreams."
    },
    {
        "title": "Knives Out (2019)",
        "expected_genre": "Mystery",
        "plot": "A detective investigates the death of a wealthy patriarch of an eccentric, combative family inside a grand mansion where every suspect has a hidden motive."
    },
    {
        "title": "Zootopia (2016)",
        "expected_genre": "Animation",
        "plot": "In a city of anthropomorphic animals, a rookie bunny cop and a cynical con artist fox must work together to uncover a conspiracy behind missing predator mammals."
    },
    {
        "title": "Se7en (1995)",
        "expected_genre": "Thriller",
        "plot": "Two homicide detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives in a dark rain-soaked metropolis."
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    model_name = data.get("model", "Logistic Regression")

    if not text or not text.strip():
        return jsonify({"error": "Please enter a valid movie plot summary."}), 400

    result = predictor.predict(text, model_name=model_name)
    return jsonify(result)

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(get_latest_metrics())

@app.route("/api/samples", methods=["GET"])
def api_samples():
    return jsonify(SAMPLE_PLOTS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
