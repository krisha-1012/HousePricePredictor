from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])

    input_data = np.array([[area, bedrooms]])

    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ₹ {prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
