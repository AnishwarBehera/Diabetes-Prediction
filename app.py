from flask import Flask,render_template,request,redirect,url_for
import numpy as np
import pandas as pd
from src.pipeline.preediction_pipeline import PredictionPipeline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_diabetes():
    gender = request.form.get('gender')
    age = int(request.form.get('age'))
    hypertension = float(request.form.get('hypertension'))
    heart_disease = float(request.form.get('heart_disease'))
    bmi = float(request.form.get('bmi'))
    Hba1c_level = float(request.form.get('HbA1c_level'))
    blood_glucose_level = float(request.form.get('blood_glucose_level'))

    final_data=pd.DataFrame([[gender,age,hypertension,heart_disease,bmi,Hba1c_level,blood_glucose_level]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
    pipeline_ = PredictionPipeline()
    p = pipeline_.predict(final_data)
    result=p
    if p[0]>0.5:
        result='There is a high probability that the patent suffers from diabetes'
    else:
        result='There is a low probability that the patent will suffer from diabetes'
    
    return render_template('result.html',result=result)

if __name__=='__main__':
    # app.run(host="0.0.0.0")
    app.run(debug=True)
    

# Test----->
#-------------------
#['Male',80,0,1,28.42,66.2,145]-Yes
#['Female',80,0,1,25.19,6.6,140]-No
