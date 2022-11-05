import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    # if data.values then we will get dictionary values 
    # then we are transforming it into an list then we will transfer it into  an array an 
    # and then we will provide this as a data point 
    # a single record with so many features which is based on our dataset 
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    # transformed or standardized 
    output=regmodel.predict(new_data)
    print(output[0])
     # this will be a 2d array so we want the first value ie 0 
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    # transformation or standardization is necessary for prediction purpose 
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Predicted House Price is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     