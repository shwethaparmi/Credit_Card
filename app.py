from flask import Flask, render_template, request, send_file
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
from src.pipeline.prediction_pipeline import ModelPredictor
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')
# ... Your other routes and configurations ...

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file path from the request
        test_csv_file = request.files['file_path']

        # Specify the path for the trained model file
        trained_model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

        # Create an instance of ModelPredictor
        model_predictor = ModelPredictor(trained_model_path,preprocessor_path)

        # Make predictions and save them to an Excel file in the "artifacts" directory
        prediction_excel_path = model_predictor.save_predictions_to_excel(test_csv_file)
        print(prediction_excel_path)

        # Send the prediction Excel file as a response for download
        return send_file(prediction_excel_path, as_attachment=True,download_name='predict.xlsx')

    except CustomException as e:
        logging.error(f'Error: {e}')
        return f"An error occurred: {e}"
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
