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


def predict(file_path, model_path):
    try:
        model = load_model(model_path)
        lb = pickle.load(open('encoder.pkl', 'rb'))
        features = [get_features(file_path)]
        df = pd.DataFrame.from_dict({'Path':file_path, 'Features':features})
        X_predict = np.array(df.Features.tolist())
        coded_predictions = model.predict_classes(X_predict)
        return lb.inverse_transform(coded_predictions)
    except Exception as e:
        print(e)

def get_features(file_path, num_features=40, res_type='kaiser_fast'):

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
