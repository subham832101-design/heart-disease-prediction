from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model_pipe.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Numerical Features
        Age = int(request.form["Age"])
        RestingBP = int(request.form["RestingBP"])
        Cholesterol = int(request.form["Cholesterol"])
        FastingBS = int(request.form["FastingBS"])
        MaxHR = int(request.form["MaxHR"])
        Oldpeak = float(request.form["Oldpeak"])

        # One-Hot Encoded Features
        Sex_M = int(request.form["Sex_M"])
        ChestPainType_ATA = int(request.form["ChestPainType_ATA"])
        ChestPainType_NAP = int(request.form["ChestPainType_NAP"])
        ChestPainType_TA = int(request.form["ChestPainType_TA"])
        RestingECG_Normal = int(request.form["RestingECG_Normal"])
        RestingECG_ST = int(request.form["RestingECG_ST"])
        ExerciseAngina_Y = int(request.form["ExerciseAngina_Y"])
        ST_Slope_Flat = int(request.form["ST_Slope_Flat"])
        ST_Slope_Up = int(request.form["ST_Slope_Up"])

        # Create DataFrame
        input_data = pd.DataFrame([[
            Age,
            RestingBP,
            Cholesterol,
            FastingBS,
            MaxHR,
            Oldpeak,
            Sex_M,
            ChestPainType_ATA,
            ChestPainType_NAP,
            ChestPainType_TA,
            RestingECG_Normal,
            RestingECG_ST,
            ExerciseAngina_Y,
            ST_Slope_Flat,
            ST_Slope_Up
        ]], columns=[
            "Age",
            "RestingBP",
            "Cholesterol",
            "FastingBS",
            "MaxHR",
            "Oldpeak",
            "Sex_M",
            "ChestPainType_ATA",
            "ChestPainType_NAP",
            "ChestPainType_TA",
            "RestingECG_Normal",
            "RestingECG_ST",
            "ExerciseAngina_Y",
            "ST_Slope_Flat",
            "ST_Slope_Up"
        ])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "⚠️ High Risk of Heart Disease"
        else:
            result = "✅ Low Risk of Heart Disease"

        return render_template(
            "index.html",
            prediction_text=f"Prediction Result: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)