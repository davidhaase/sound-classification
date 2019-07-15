from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from model_generator import get_3_log_model
from multifileprocess import data_label_load
import numpy as np

def fit_dict(data,labels,model,keys,**kwargs):
    '''
    To be used after the data has been processed.
    Fits the model to the processed data.

    Parameters:
        data
            the data dictionary produced by multifileprocess
        labels
            the label list produced by multifileprocess
        model
            Keras multiinput model
        keys
            the ordered list of keys for specifying which
            data features to pass to the model
        **kwargs
            will be passed to model.fit()
            Ex/ epochs = 100
    '''
    input_vals = [reshape_for_model(data[key]) for key in keys]
    labs = to_categorical(LabelEncoder().fit_transform(labels))
    return model.fit(x=input_vals,y=labs,**kwargs)
def reshape_for_model(data):
    return data.reshape(tuple(list(data.shape)+[1]))

def concate_multiprocess_outputs(data_list,labels_list):
    """
    Concatonates a list of data dictionaries and a list of labels
    that were processed seperately.

    Parameters
        data_list
            The list of data dictionaries [must be same shapes]
        labels_list
            List of labels lists
    Returns
        Concatonated data, Concatonated labels
    """
    step = 0
    full_labels = []
    for labels in labels_list:
        if step%3 == 0:

            print(f"step is {step}",end="\r")
        step+=1
        full_labels = full_labels + labels
    full_data = data_list[0]
    step = 0
    for other_data in data_list[1:]:
        for key in full_data.keys():
            if step%3 == 0:
                print(f"step is {step}",end="\r")
            step+=1
            full_data[key] = np.append(full_data[key],other_data[key],axis=0)
    return full_data,full_labels

def list_of_pickle_paths_to_data_list_and_labels(paths):
    data_list = []
    label_list = []
    for path in paths:
        dat,lab = data_label_load(path)
        data_list.append(dat)
        label_list.append(lab)
    return data_list,label_list
import os
def get_all_paths(dir_path):
    all_paths = []
    for file_name in os.listdir(dir_path):
        all_paths.append(dir_path+file_name)
    return all_paths
