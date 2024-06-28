from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from ImageClassifier.utils.common import decodeImage
from ImageClassifier.pipeline.predict import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing (CORS) for all routes to allow requests from any domain


class ClientApp:
    def __init__(self):
        ## the input will be stored here when using decodeImage function
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# Run the Html Page
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


# Run the Prediction Script
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    #app.run(host='0.0.0.0', port=80) #for AZURE