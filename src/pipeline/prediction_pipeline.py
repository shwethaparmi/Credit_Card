import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class ModelPredictor:
    def __init__(self, trained_model_path, preprocessor_path):
        self.trained_model_path = trained_model_path
        self.preprocessor_path = preprocessor_path

    def load_trained_model(self):
        try:
            model = load_object(self.trained_model_path)
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def load_preprocessor(self):
        try:
            preprocessor = load_object(self.preprocessor_path)
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_test_data(self, test_csv_file):
        try:
            # Load the preprocessor
            preprocessor = self.load_preprocessor()

            # Load the test data from the CSV file into a DataFrame
            test_data = pd.read_csv(test_csv_file)

            # Apply the same preprocessing used during training
            preprocessed_data = preprocessor.transform(test_data)

            return preprocessed_data

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, test_csv_file):
        try:
            # Load the trained model
            model = self.load_trained_model()

            # Preprocess the test data
            preprocessed_data = self.preprocess_test_data(test_csv_file)

            # Make predictions on the preprocessed test data
            predictions = model.predict(preprocessed_data)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)

    def save_predictions_to_excel(self, test_csv_file):
        try:
            # Make predictions on the test data
            predictions = self.predict(test_csv_file)

            # Create a DataFrame with the predictions
            prediction_df = pd.DataFrame({'Predicted_Label': predictions})

            # Specify the path for the output Excel file in the "artifacts" directory
            output_excel_file = os.path.join('artifacts', 'predictions.xlsx')

            # Save the predictions to the Excel file
            prediction_df.to_excel(output_excel_file, index=True)

        except Exception as e:
            raise CustomException(e, sys)
        return output_excel_file

if __name__ == "__main__":
    # Specify the paths for the test CSV file, trained model file, and preprocessor object
    test_csv_file = os.path.join('artifacts', 'test.csv')
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    # Create an instance of ModelPredictor
    model_predictor = ModelPredictor(trained_model_path, preprocessor_path)

    # Make predictions and save them to an Excel file in the "artifacts" directory
    model_predictor.save_predictions_to_excel(test_csv_file)
