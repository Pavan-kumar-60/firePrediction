from flask import Flask, render_template, request, jsonify
import pandas as pd 
import numpy as np
import pickle


application = Flask(__name__)
app = application


ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scalar_model = pickle.load(open('models/scalar.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__=='__main__':
    app.run(debug=True)