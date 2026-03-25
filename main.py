from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Home Page
@app.route('/')
def home():
    return render_template("index.html")

# Contact Page
@app.route('/contact', methods=["GET"])
def contact_page():
    return render_template("contact.html")

@app.route('/history', methods=["GET"])
def history_page():
    return render_template("history.html")

# Predict Page (GET → open form)
@app.route('/predict', methods=["GET"])
def predict_page():
    return render_template("predict.html")

# Predict Logic (POST → form submit)
@app.route('/predict', methods=["POST"])
def predict():
    try:
        features = [
            int(request.form.get("gender", 0)),
            int(request.form.get("SeniorCitizen", 0)),
            int(request.form.get("Partner", 0)),
            int(request.form.get("Dependents", 0)),
            int(request.form.get("tenure", 0)),
            int(request.form.get("PhoneService", 0)),
            int(request.form.get("MultipleLines", 0)),
            int(request.form.get("InternetService", 0)),
            int(request.form.get("OnlineSecurity", 0)),
            int(request.form.get("OnlineBackup", 0)),
            int(request.form.get("DeviceProtection", 0)),
            int(request.form.get("TechSupport", 0)),
            int(request.form.get("StreamingTV", 0)),
            int(request.form.get("StreamingMovies", 0)),
            int(request.form.get("Contract", 0)),
            int(request.form.get("PaperlessBilling", 0)),
            int(request.form.get("PaymentMethod", 0)),
            float(request.form.get("MonthlyCharges", 0)),
            float(request.form.get("TotalCharges", 0))
        ]

        # Convert to numpy
        data = np.array([features])

        # Scale
        data = scaler.transform(data)

        # Predict
        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Customer will Churn ❌"
        else:
            result = "Customer will Stay ✅"

        # Show result on predict page (better UX)
        return render_template("predict.html", prediction_text=result)

    except Exception as e:
        return render_template("predict.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)