from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)


with open("iq.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
           
            features = [
                float(request.form.get("reaction_time")),
                float(request.form.get("education_years")),
                float(request.form.get("age")),
                float(request.form.get("gpa")),
                float(request.form.get("break_time")),
            ]
            
           
            scaled = scaler.transform([features])
            prediction = model.predict(scaled)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
