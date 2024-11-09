from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS 
import numpy as np
from math import isnan
app = Flask(__name__)
# Applying CORS to all routes with specific origin
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Load your trained model (e.g., using pickle or joblib)
with open('xgb_regressor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the React frontend
        data = request.get_json()
        
        # Validate and convert input values to numbers (check for type issues)
        features = [
            float(data['bedrooms']),
            float(data['bathrooms']),
            float(data['sqftLiving']),
            float(data['sqftLot']),
            float(data['floors']),
            float(data['waterfront']),
            float(data['view']),
            float(data['condition']),
            float(data['grade']),
            float(data['sqftAbove']),
            float(data['sqftBasement']),
            float(data['yearBuilt']),
            float(data['yearRenovated']),
            float(data['zipcode'])
        ]
        
        # Check for invalid data
        if any(isnan(val) or val is None for val in features):
            return jsonify({'error': 'Invalid or missing input data'}), 400

        # Reshape and predict with the model
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({'prediction': float(prediction[0])})

    except ValueError as e:
        return jsonify({'error': f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
