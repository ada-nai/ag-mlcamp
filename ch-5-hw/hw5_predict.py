import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model2.bin'
dict_file = 'dv.bin'
threshold = 0.5

with open(model_file, 'rb') as infile:
    model = pickle.load(infile)

with open(dict_file, 'rb') as infile:
    dv = pickle.load(infile)

app = Flask('hw')

@app.route('/predict', methods= ['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0][1]
    churn = y_pred >= threshold

    result = {
        'churn_probability' : float(y_pred),
        'churn' : bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug= True, host= '0.0.0.0', port= 5501)
