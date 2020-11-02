# Speech Measure Application - Server Side

This is part of speech rate measure project.
This is the server side of web-application which is part of ML project to measure Speech Rate.

## Installation instructions
For windows - you can install the file [here](https://github.com/almog1/SpeechMeasureServer/tree/main/serverInstaller), and after installation complete, run the file: "run_speech_server.bat"

or 

You can run the Server.py, installation needed:
* Pytorch 1.7+
* librosa
* flask
* flask_cors
* soundfile
* pydub
* shutil
* wave


## Client
Follow the instructions [here](https://github.com/Jenny-Smolenksy/speech-rate-client) to load the client

## How it works?

The server is written in python using Flask
It works on port 8000 and get post request from the [client](https://github.com/Jenny-Smolenksy/speech-rate-client)

It use a model to predict SPS (Syllable Per Second) in the audio file from the request

You are able to change the model by changing in modelTest argumnets from 'YOLO' to 'CLASSIFICATION'
For farther documentation see:
* [Multiclass model](https://github.com/Jenny-Smolenksy/ClassificationSpeechNet)
* [Hybrid Yolo Model](https://github.com/almog1/SpeechVowelsNet)



## Authors

**Jenny Smolensky** , **Almog Gueta**
