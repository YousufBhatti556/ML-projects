import os, sys
from flask import Flask, request, render_template
from mlproject.pipeline.predict_pipeline import CustomData, PredictPipeline
from mlproject.exception import CustomException
from mlproject.logger import logging

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score"))
        )

        final_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict_data(final_df)

        return render_template('home.html', prediction=round(pred[0], 2))

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('home.html', error=f"Prediction Error: {e}")
if __name__ == '__main__':
    app.run(debug=True, port=5001)
