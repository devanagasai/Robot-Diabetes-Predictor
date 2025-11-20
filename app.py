from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Create a Flask app instance
app = Flask(__name__)

# --- Load the model and encoders ---
# Load the pre-trained model
model = joblib.load('dt_tuned_model.pkl')

# Load the label encoders
gender_le = joblib.load('gender_le.pkl')
smoking_le = joblib.load('smoking_history_le.pkl')

# --- Main API endpoint ---
@app.route('/')
def home():
    """Renders the HTML form for user input."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, preprocesses it, and returns a prediction."""
    try:
        # Get data from the POST request
        data = request.get_json(force=True)

        # Convert the data into a DataFrame
        input_data = pd.DataFrame([data])

        # Preprocess the categorical features using the loaded encoders
        input_data['gender'] = gender_le.transform(input_data['gender'])
        input_data['smoking_history'] = smoking_le.transform(input_data['smoking_history'])

        # Reorder the columns to match the model's training data
        column_order = [
            'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
            'bmi', 'HbA1c_level', 'blood_glucose_level'
        ]
        input_data = input_data[column_order]

        # Make a prediction
        prediction = model.predict(input_data)

        # Convert prediction to human-readable format
        if prediction[0] == 1:
            result = "Positive for Diabetes"
        else:
            result = "Negative for Diabetes"

        # Return the prediction as a JSON response
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})