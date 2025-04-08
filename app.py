
# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open("data/model.pkl", "rb"))

# @app.route("/")  # Main page route
# def home():
#     return render_template("home.html",prediction = None)  # Renders home page

# @app.route("/predict_page")  # Route to prediction page
# def predict_page():
#     return render_template("p.html" , prediction = None)

# @app.route("/predict", methods=["POST"])  # Prediction logic
# def predict():
#     try:
#         # Get user inputs from form
#         features = [
#             request.form["age_band"],
#             request.form["vehicle_type"],
#             request.form["road_surface"],
#             request.form["light_conditions"],
#             request.form["weather"],
#             request.form["collision_type"],
#             request.form["casualties"],
#             request.form["pedestrian_movement"],
#             request.form["hour"],
#             request.form["day_of_week"],
#             request.form["cause_of_accident"],
#             request.form["gender"]
#         ]

#         # Convert input data into NumPy array
#         features_array = np.array(features).reshape(1, -1)

#         # Make prediction
#         # prediction = model.predict(features_array)[0]
#         prediction = int(model.predict(features_array)[0])

#         print("Predicted class:", prediction)


#         return render_template("p.html", prediction=prediction)

#     except Exception as e:
#         return render_template("p.html", prediction=f"Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True)
    

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("data/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_page")
def predict_page():
    return render_template("p.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from form
        features = [
            request.form["age_band"],
            request.form["vehicle_type"],
            request.form["road_surface"],
            request.form["light_conditions"],
            request.form["weather"],
            request.form["collision_type"],
            request.form["casualties"],
            request.form["pedestrian_movement"],
            request.form["hour"],
            request.form["day_of_week"],
            request.form["cause_of_accident"],
            request.form["gender"]
        ]

        # Convert input data into NumPy array
        features_array = np.array(features).reshape(1, -1)

        print("Input features from form:", features_array)

        # Make prediction
        prediction = int(model.predict(features_array)[0])

        # Mapping prediction to labels and descriptions
        severity_map = {
            0: ("Fatal", "🔴 Fatal injury – Life-threatening or resulted in death."),
            1: ("Serious", "🟠 Serious injury – Requires medical treatment and possibly hospitalization."),
            2: ("Slight", "🟢 Slight injury – Minor injuries with no immediate threat to life.")
        }

        severity_label, severity_description = severity_map.get(
            prediction, ("Unknown", "Unable to determine severity from the input.")
        )

        return render_template("result.html", severity_label=severity_label, severity_description=severity_description)

    except Exception as e:
        return render_template("result.html", severity_label="Error", severity_description=str(e))

if __name__ == "__main__":
    app.run(debug=True)