import os
from flask import Flask, request, session
from flask import jsonify
from time import sleep
from SpeechRateTest_forModel import main as modelTest
from gevent.pywsgi import WSGIServer

from werkzeug.utils import secure_filename
from flask_cors import CORS

# create app
app = Flask(__name__)

#Post- get request from client with audio file
# send it to the model that return prediction of SPS
@app.route('/upload', methods=['POST'])
def fileUpload():
    target = ".\\uploadedFilesDir"
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = secure_filename(file.filename)

    response = {
        'Message': 'fail',
        'Failure': '',
        'Result': 0
    }
    #check if file name end with the allowed etensions
    if filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        response['Message'] = 'failed'
        response['Failure'] = 'Upload failed! File format is not allowed'
        return jsonify(response)

    destination = "/".join([target, filename])
    file.save(destination)
    session['uploadFilePath'] = destination


    try:
        # run model - get file path, model name : 'YOLO' or 'CLASSIFICATION' , t if chosen yolo
        sps_predicted = modelTest(destination, 'YOLO', 0.25)
    except:
        print("failed")
        response['Message'] = 'failed'
        response['Failure'] = 'Upload failed! Model failed to upload file'
    else:
        #if the function run without errors
        response['Message'] = 'Succeeded'
        response['Result'] = 'SPS: ' + str(sps_predicted)

    return jsonify(response)

if __name__ == '__main__':
    ALLOWED_EXTENSIONS = set(['flac', 'wav', 'mp3','webm'])
   # UPLOADED_FILES_ALLOW
    CORS(app, expose_headers='Authorization')
    app.secret_key = os.urandom(24)

    http_server = WSGIServer(('', 8000), app)
    http_server.serve_forever()
