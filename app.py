#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle5 as pickle
import numpy as np

app = Flask(__name__)
predictModel = pickle.load(open("Stress_model_save (1)","rb"))
scaleModel = pickle.load(open("scaleModel","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [int(x) for x in request.form.values()]
    scaledValue = scaleModel.transform([int_features]) 
    loadModel = pickle.load(open("Stress_model_save (1)","rb"))
    result = loadModel.predict(scaledValue)    
   
    return render_template('index.html', pred='Student passing probability is :  {}'.format(result[0]))

if __name__ == '__main__':
    app.run(debug=False)
