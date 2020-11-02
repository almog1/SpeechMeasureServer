import os
from pathlib import Path
import librosa
from pydub import AudioSegment
import subprocess
from subprocess import Popen,PIPE
import shutil

import wave
from pydub import AudioSegment as am

import soundfile as sf
from neural_net.NeuralNet import NeuralNet
from yolo_net.Yolo import Yolo
import warnings
warnings.filterwarnings("ignore")

def wav_splitter(original_wav, start, end, destination_file):
    audio_file = AudioSegment.from_file(original_wav)
    audio_file = audio_file[start:end]
    audio_file.export(destination_file, format="wav")  # Exports to a wav file in the current path.


def silence_padding_func(original_wav, start, end, padding, destination_file):
    audio_file = AudioSegment.from_file(original_wav)
    audio_file = audio_file[start:end]
    silence = AudioSegment.silent(duration=padding)
    audio_file = audio_file + silence
    audio_file.export(destination_file, format="wav")  # Exports to a wav file in the current path.


def cut_file_by_interval(audio_file, dst_folder, interval=0.5):
    end_of_file = librosa.get_duration(filename=audio_file) * 1000

    start_interval = 0
    end_interval = 500
    silence_padding = 0
    index = 0
    Path(dst_folder + "\\temp").mkdir(parents=True, exist_ok=True)
    while start_interval < end_of_file:

        dst_file = dst_folder + "\\temp\\" + os.path.basename(audio_file).replace(".wav", str(index)) + ".wav"
        index += 1

        if silence_padding <= 0:
            wav_splitter(audio_file, start_interval, end_interval, dst_file)
        else:
            silence_padding_func(audio_file, start_interval, end_of_file, silence_padding, dst_file)

        start_interval += 500
        end_interval += 500

        silence_padding = end_interval - end_of_file

# convert file from flac to wav
def convert_flac(file_path):
    dest_path_current = file_path.replace("flac", "wav")
    # convert to wav
    data, sample_rate = sf.read(file_path)
    sf.write(dest_path_current, data, sample_rate, format="wav")

    return dest_path_current


# convert file from .webm to wav
def convert_webm(file_path):
    original_file = file_path
    new_path = file_path.replace(".webm", ".wav")
    # create the line command of converting webm to wav and run it
    command = "ffmpeg\\bin\\ffmpeg.exe -i \"" + original_file + "\" -y \"" + new_path + "\""
    p = subprocess.check_call(command,shell=True)
    return new_path


# get file path and create it in correct format
def convert_file(file_path):
    # if no needed - stay .wav
    new_path = file_path
    if file_path.endswith('.flac'):
        # convert from .flac
        new_path = convert_flac(file_path)
    if file_path.endswith('.webm'):
        # convert from .webm
        new_path = convert_webm(file_path)

    return new_path

#convert file fr,channels
def change_file_properties(audio_file):
    with wave.open(audio_file, "rb") as wave_file:
        frame_rate = wave_file.getframerate()

    # set the frame rate to 16K
    sound = am.from_file(audio_file, format='wav', frame_rate=frame_rate)
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)

    new_full_path = audio_file.replace(".wav", "_for_split.wav")
    # write file to destination
    sound.export(new_full_path, format='wav')
    return new_full_path

def cut_files_to_folder(new_full_path):
    #delete folder if it exist
    destination_path = os.getcwd() + "\\temp"
    if os.path.exists(destination_path) and os.path.isdir(destination_path):
        shutil.rmtree(destination_path)

    Path(destination_path).mkdir(parents=True, exist_ok=True)
    cut_file_by_interval(new_full_path, destination_path)
    return destination_path

# The main get the file need to be predicted
def main(file_path,model_name,threshold=0.25):
    sps = 0

    # convert file format
    audio_file = convert_file(file_path)
    # change fr,channels such that model can read it
    new_full_path = change_file_properties(audio_file)
    # cut the file to 500ms each and run model
    destination_path = cut_files_to_folder(new_full_path)

    # path and load the model
    if model_name == 'YOLO':
        model_path = "trained\yolo_hybrid_model.pth"
        net = Yolo(model_path, is_extended=True)
        # threshold value for the model confidence
        t = threshold
        # run the model to get the prediction
        prediction = net.create_new_test(destination_path, t)
    elif model_name == 'CLASSIFICATION':
        model_path = "trained\multiclass_model.pth"
        net = NeuralNet(model_path, 7)
        prediction = net.create_new_test(destination_path)
    else:
        return "Error in model name!"

    duration = librosa.get_duration(filename=audio_file)
    sps = prediction / duration

    print(f"{prediction}->SPS:{sps} ")

    #delete all the files created here
    shutil.rmtree(destination_path)
    #check if created new
    if audio_file != file_path:
        os.remove(file_path)
    os.remove(audio_file)
    os.remove(new_full_path)

    sps = round(sps, 2)

    return sps

if __name__ == '__main__':
    main()
