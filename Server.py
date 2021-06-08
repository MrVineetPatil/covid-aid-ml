import numpy as np
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/', methods=['POST'])

def predict():
    # Get the data from the POST request.
   # data = request.args.get(force=True)
    # Make prediction using model loaded from disk as per the data.
    # features = ['Contact with confirmed', 'Headache', 'Sore throat', 'Shortness of breath', 'Cough', 'Fever', 'Male', 'Age 60+']
    prediction = model.predict([np.array([request.args.get('Contact with confirmed'),
                                          request.args.get('Headache'),
                                          request.args.get('Sore throat'),
                                          request.args.get('Shortness of breath'),
                                          request.args.get('Cough'),
                                          request.args.get('Fever'),
                                          request.args.get('Male'),
                                          request.args.get('Age 60+')])])
    # Take the first value of prediction
    output = prediction[0]
    print([np.array([request.args.get('Contact with confirmed'),
                                          request.args.get('Headache'),
                                          request.args.get('Sore throat'),
                                          request.args.get('Shortness of breath'),
                                          request.args.get('Cough'),
                                          request.args.get('Fever'),
                                          request.args.get('Male'),
                                          request.args.get('Age')])])
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='192.168.0.166',debug=True)
# #