"""
Smart Lender -- Flask web app

Loads the trained model from model/model.pkl and serves a form where an
applicant's details can be submitted for an instant approval prediction.

Run:  python app.py
Then open http://localhost:5000
"""
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

with open("model/model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
model_name = bundle["model_name"]
uses_scaled_input = bundle["uses_scaled_input"]
scaler = bundle["scaler"]
encoders = bundle["encoders"]
target_encoder = bundle["target_encoder"]
feature_cols = bundle["feature_cols"]

FORM_FIELDS = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "Property_Area": ["Urban", "Semiurban", "Rural"],
}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", fields=FORM_FIELDS, model_name=model_name, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    try:
        row = {
            "Gender": form["Gender"],
            "Married": form["Married"],
            "Dependents": form["Dependents"],
            "Education": form["Education"],
            "Self_Employed": form["Self_Employed"],
            "Property_Area": form["Property_Area"],
            "ApplicantIncome": float(form["ApplicantIncome"]),
            "CoapplicantIncome": float(form["CoapplicantIncome"]),
            "LoanAmount": float(form["LoanAmount"]),
            "Loan_Amount_Term": float(form["Loan_Amount_Term"]),
            "Credit_History": float(form["Credit_History"]),
        }
    except (KeyError, ValueError):
        return render_template(
            "index.html", fields=FORM_FIELDS, model_name=model_name,
            result=None, error="Please fill in every field with valid values.",
        )

    encoded = []
    for col in feature_cols:
        if col in encoders:
            le = encoders[col]
            encoded.append(le.transform([row[col]])[0])
        else:
            encoded.append(row[col])
    X = np.array(encoded).reshape(1, -1)

    if uses_scaled_input:
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    label = target_encoder.inverse_transform([pred])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(np.max(model.predict_proba(X))) * 100, 1)

    approved = label == "Y"

    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        model_name=model_name,
        result={"approved": approved, "confidence": proba},
        form_values=form,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
