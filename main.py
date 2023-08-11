from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("lgbm_model_selected_features.pkl")  # Update with your model filename

THRESHOLD = 0.5


@app.route('/predict/', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data from the request

    # Define the columns in the same order as used during training
    columns = ['ORGANIZATION_TYPE','EXT_SOURCE_AVG','OCCUPATION_TYPE', 'AMT_GOODS_PRICE', 'PREV_REFUSED', 'AMT_ANNUITY',
               'AMT_CREDIT', 'PREV_APPROVED', 'AGE', 'CREDITS_ACTIVE']

    # Create a DataFrame from the JSON data
    df = pd.DataFrame([data], columns=columns)

    # Extract the features from the DataFrame
    features = df.values.tolist()[0]

    # Predict using the model
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0][1]

    output = {'prediction': int(prediction[0]), 'probability': probability}
    return jsonify(output)


@app.route('/')
def index():
    return 'Welcome to the Loan Status API'


if __name__ == "__main__":
    app.run(debug=True)

