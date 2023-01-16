## Load packages
import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

## Create a basic Flask app
app=Flask(__name__)

## Load the pickled model
regmodel=pickle.load(
    open(
        'regression_pipeline.pkl',
        'rb'
    )
)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output=np.round(regmodel.predict(new_data), 2)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)
   
     