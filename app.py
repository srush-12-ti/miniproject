from flask import Flask, request, jsonify, make_response
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the machine learning model
model = joblib.load("./ensemble_model_binary_compressed_1.joblib")  # Update path if necessary

# Function to convert input data columns to appropriate types
def convert_input_data(data):
    # Convert categorical columns to numeric (if they are encoded as strings)
    categorical_columns = ['HighChol', 'BMI', 'Diabetes', 'PhysActivity', 'HvyAlcoholConsump',
                           'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Education',
                           'Current_Smoker', 'Income_Category', 'Age_Group', 'On_BP_Medication', 'Fruits_Veggies']

    # Ensure all categorical columns are converted to numeric
    for col in categorical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce', downcast='integer')  # Coerce errors to NaN if non-numeric

    return data

@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight OPTIONS request
        response = make_response(jsonify({}))
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    if request.method == 'POST':
        # Handle the POST request
        data = request.get_json()

        # Convert input data into DataFrame
        input_data = pd.DataFrame([{
            'HighChol': data['HighChol'],
            'BMI': data['BMI'],
            'Diabetes': data['Diabetes'],
            'PhysActivity': data['PhysActivity'],
            'HvyAlcoholConsump': data['HvyAlcoholConsump'],
            'GenHlth': data['GenHlth'],
            'MentHlth': data['MentHlth'],
            'PhysHlth': data['PhysHlth'],
            'DiffWalk': data['DiffWalk'],
            'Sex': data['Sex'],
            'Education': data['Education'],
            'Current_Smoker': data['Current_Smoker'],
            'Income_Category': data['Income_Category'],
            'Age_Group': data['Age_Group'],
            'On_BP_Medication': data['On_BP_Medication'],
            'Fruits_Veggies': data['Fruits_Veggies'],
        }])

        # Convert input data types
        input_data = convert_input_data(input_data)

        # Make prediction using the trained modelser
        prediction = model.predict_proba(input_data)
        positive_class_probability = prediction[0][1]

        # Return prediction result as JSON
        response = jsonify({"prediction": float(positive_class_probability)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
