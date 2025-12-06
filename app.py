from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("spam_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    prediction = model.predict([message])[0]  # spam / ham
    return render_template("index.html", result=prediction, message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

