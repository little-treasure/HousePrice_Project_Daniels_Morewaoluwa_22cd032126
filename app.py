from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {
            "OverallQual": int(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "GarageCars": int(request.form["GarageCars"]),
            "BedroomAbvGr": int(request.form["BedroomAbvGr"]),
            "YearBuilt": int(request.form["YearBuilt"])
        }

        df = pd.DataFrame([data])
        price = model.predict(df)[0]
        prediction = f"${price:,.2f}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()