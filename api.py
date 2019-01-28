from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # If model exists, then predict
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            pred = list(lr.predict(query))
            return (jsonify({'prediction': str(pred)}))
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # default port

    lr = joblib.load("model.pkl") # Load model
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load training data
    print ('Model columns loaded')

    app.run(port=port, debug=True)