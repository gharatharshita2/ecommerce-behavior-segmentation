import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURES = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
            'TotalPages', 'TotalDuration', 'ProductFocusRatio']

@app.route('/')
def home():
    return "E-commerce conversion prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        input_values = [data[feature] for feature in FEATURES]
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400

    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        'will_purchase': bool(prediction),
        'purchase_probability': round(float(probability), 3)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)