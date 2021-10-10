import pickle
from os import environ
from flask import Flask
from flask import request
from flask import jsonify

dv_file = (environ.get("FILE_PATH") or './data/models/') + 'dv.bin'
model_file = (environ.get("FILE_PATH") or './data/models/') + 'model1.bin'

model = None
dv = None

app = Flask('homework-05')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


@app.before_first_request
def load_models():
    global dv_file, model_file
    print(dv_file, model_file)
    global model, dv

    with open(model_file, 'rb') as f_in:
        model = pickle.load(f_in)

    with open(dv_file, 'rb') as f_in:
        dv = pickle.load(f_in)
    
    print("LOADED MODELS")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1234)