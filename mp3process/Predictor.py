import sounddevice as sd
from scipy.io.wavfile import write
import sys
import os
import time
sample_rate = 44100
seconds_per_audio_file = 20
amount_of_channels = 1
relative_path_temp_audio = "temp/temp.wav"
relative_path_to_code = "code/predictor_function.py"
relative_path_to_model = "model/keras_model"
welcome_message = f"""
Welcome to David and Noah's language classifier.
After pressing return you will be recorded for {seconds_per_audio_file}.
Please speak in Russian, Arabic, Greek, Turkish or Farsi.
Accent and fluency do matter, so only use languages you are
either native or near native.
"""

def start():
    path_to_model = sys.argv[1]
    path_to_src = sys.argv[2]

    temp_audio_path = append_path(path_to_src + relative_path_temp_audio)
    print(welcome_message)
    print("(Press return to record)           ",end='\r')
    input()
    print("recording...                       ",end="\r")
    record(sample_rate,seconds_per_audio_file,append_path(path_to_src+relative_path_temp_audio),amount_of_channels)

    sys.path.append(append_path(path_to_src,relative_path_to_code))
    from predictor_function import *
    model = load_model(append_path(path_to_src,relative_path_to_model))

    prediction = predict(append_path(path_to_src+relative_path_temp_audio),model)

    print("prediction is                      ",end="\r")
    sleep_time = .1
    time.sleep(sleep_time)
    print("prediction is.                     ",end="\r")
    time.sleep(sleep_time)
    print("prediction is..                    ",end="\r")
    time.sleep(sleep_time)
    print("prediction is...                   ",end="\r")
    time.sleep(1)
    print(f"{prediction}!                     ")
    print("Try again?                         ",end="\r")
    start()











def record(sample_rate,seconds,output_filename,channels):
    fs = sample_rate  # Sample rate
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels)
    sd.wait()  # Wait until recording is finished
    write('myfile.wav', fs, myrecording)  # Save as WAV file
def append_path(base, path):
    if base[-1] != '/':
        base += '/'
    if path[0] == "/":
        path = path[1:]
    return base+path










if _name_ == "_main_":
    start()
