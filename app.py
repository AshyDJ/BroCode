from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS 
import numpy as np
from math import isnan
app = Flask(__name__)
# Applying CORS to all routes with specific origin
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Load your trained model (e.g., using pickle or joblib)
with open('xgb_regressor_model.pkl', 'rb') as model_file1:
    xgb_model = pickle.load(model_file1)

with open('GBR.pkl', 'rb') as model_file2:
    gbr_model = pickle.load(model_file2)

with open('linear_regression_model.pkl', 'rb') as model_file3:
    lgr_model = pickle.load(model_file3)



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
        prediction1 = xgb_model.predict(features)

        prediction2 = gbr_model.predict(features)

        prediction3 = lgr_model.predict(features)
        


        def maxandmin():
            print('gello')
            array=[prediction1[0],prediction2[0],prediction3[0]]
            finalmax=max(array)
            finalmin=min(array)

            return finalmax,finalmin
        

        maximum,minimum=maxandmin()


        return jsonify({
            'prediction1': round(float(minimum), 2),
            'prediction2': round(float(maximum), 2)
        })
    except ValueError as e:
        return jsonify({'error': f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
