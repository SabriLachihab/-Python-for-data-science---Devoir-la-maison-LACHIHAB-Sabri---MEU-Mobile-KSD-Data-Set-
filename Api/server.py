# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json
from pandas.io.json import json_normalize
import numpy as np


app = Flask(__name__)

# Load the model
model = pickle.load(open('modele.ckpt','rb'))


def predict():
    X_test = pd.read_csv('data.csv')
    X_test.index = X_test['Unnamed: 0']
    X_test = X_test.drop(['Unnamed: 0'],axis=1)
    print(X_test.head())
    prediction_results = model.predict(X_test.values)
    print(prediction_results)
    return str(prediction_results)

@app.route('/',methods=['GET'])
def bienvenue():
    return 'API SABRI LACHIHAB'

@app.route('/api',methods=['GET'])
def api():
    response = predict()
    return response

if __name__ == '__main__':
    app.run(port=5001, debug=True)
