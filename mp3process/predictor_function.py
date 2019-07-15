import pickle
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

import librosa

def extract_linear_features(file_path):
    num_features = 40
    res_type = 'kaiser_fast'

    # handle exception to check if there isn't a file which is corrupted
    try:
      # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_path, res_type=res_type)
      # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_features).T,axis=0)
    except Exception as e:
        print(e, "Error encountered while parsing file: ", file_path)
        return None

    return mfccs


def predict(path_to_audio_file, model_path):
    prediction = 'Unknown'
    lb = pickle.load( open( 'encoder.pkl', 'rb' ) )
    asset = [{'Path':path_to_audio_file}]
    test_df = pd.DataFrame.from_dict(asset)

    test_df.dropna(inplace=True)


    try:
        model = load_model(model_path)
    except Exception as e:
        print(e)
        return

    try:
        test_df['Features'] = test_df['Path'].map(lambda x: extract_linear_features(x))
    except Exception as e:
        print('Failed to extract features from {}'.format(path_to_audio_file))
        print(e)

    X_predict = np.array(test_df.Features.tolist())
    coded_predictions = model.predict_classes(X_predict)
    prediction = lb.inverse_transform(coded_predictions)[0]
    return prediction
