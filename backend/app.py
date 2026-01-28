import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "house_price_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predictor")
def predictor_page():
    return render_template("predictor.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        floors = int(request.form["floors"])
        parking = int(request.form["parking"])
        house_age = int(request.form["house_age"])

        input_data = np.array([[area, bedrooms, bathrooms, floors, parking, house_age]])

        prediction = model.predict(input_data)[0]

        return render_template(
            "predictor.html",
            prediction_text=f"₹ {prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "predictor.html",
            prediction_text="Error: Please enter valid inputs"
        )

if __name__ == "__main__":
    app.run(debug=True)
