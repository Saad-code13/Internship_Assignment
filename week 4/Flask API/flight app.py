# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:17:39 2022

@author: saad
"""

import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Pickle/model.pkl', 'rb'))
oe = pickle.load(open('Pickle/encoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html', oe = oe)

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction = model.predict(final_features)
    #output = round(prediction[0], 2)

    return render_template('index.html',oe = oe, prediction_text='Predicted Flight price $ {}'.format(prediction[0]))


if __name__ == '__main__':
    app.run(port=5000)