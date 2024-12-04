from flask import Flask, render_template, request, jsonify
import pandas as pd 
import numpy as np
import pickle


application = Flask(__name__)
app = application


reg_model = pickle.load(open('models/reg.pkl', 'rb'))
scalar_model = pickle.load(open('models/std.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model', methods=['GET'])
def model():
    return render_template('model.html')

@app.route('/predict',methods=['POST'])
def predict():
        Temperature = float(request.form.get('temperature',0))
        RH = float(request.form.get('rh',0))
        WS = float(request.form.get('ws',0))
        RAIN = float(request.form.get('rain',0))
        FFMC = float(request.form.get('ffmc',0))
        DMC = float(request.form.get('dmc',0))
        ISI = float(request.form.get('isi',0))
        CLASSES = float(request.form.get('classes'))
        REGION = float(request.form.get('region'))
        new_scaled = scalar_model.transform([[Temperature, RH, WS, RAIN, FFMC, DMC, ISI,CLASSES,REGION]])
        prediction = reg_model.predict(new_scaled)
        return render_template('result.html', predictions=round(prediction[0],2))

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__=='__main__':
    app.run(debug=True)