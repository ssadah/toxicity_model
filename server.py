# Importing the libraries
from flask import Flask, abort, request, jsonify
from sklearn.externals import joblib

# Create Flask object to run
app = Flask(__name__)

# Load the model
classifier = joblib.load('classifier_light.pkl')

@app.route('/', methods=['POST'])
def predict():
	# Check for errors
    app.logger.info("{} request received from: {}".format(request.method, request.remote_addr))
    if not request.json or 'data' not in request.json:
        app.logger.error("Request has no data or request is not json, aborting")
        abort(400)
    # Load the model
    comment = [request.json['data']]
    pred = classifier.predict(comment)[0]
    if pred == 0: pred = 'non-toxic'
    else : pred = 'toxic'
    return jsonify({'result': pred})

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)