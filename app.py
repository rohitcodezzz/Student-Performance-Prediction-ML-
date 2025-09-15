from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# ----------------------
# ROUTES
# ----------------------

@app.route('/')
def index():
    """Landing page -> index.html"""
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Form to input data -> Prediction -> Show results"""
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect data from form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            study_hours=int(request.form.get('study_hours')),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:", pred_df)

        # Run prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction Output:", results)

        prediction_score = results[0]   # Assuming model returns numeric score (0-100)

        # Render back home.html with results
        return render_template(
            'home.html',
            results=prediction_score,
            prediction_score=prediction_score
        )


@app.route('/quiz')
def quiz():
    """Quiz page"""
    return render_template("quiz.html")


# ----------------------
# MAIN ENTRY
# ----------------------
if __name__ == "__main__":
    # When running directly
    app.run(host="0.0.0.0", port=5000, debug=True)
