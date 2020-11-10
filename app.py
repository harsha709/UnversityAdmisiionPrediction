import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
app = Flask(__name__)
model = pickle.load(open('UniversityAdmissionPrediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]
    x_test = sc.fit_transform(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0]
    if(output==False):
        return render_template('index.html', prediction_text='You Dont have a chance {}'.format(output))
    else:
        return render_template('index.html', prediction_text='You have a chance')
if __name__ == "__main__":
    app.run(debug=True)
