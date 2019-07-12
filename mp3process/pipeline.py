from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
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
    input_vals = [data[key] for key in keys]
    labs = to_categorical(LabelEncoder().fit_transform(labels))
    return model.fit(x=input_vals,y=labs,**kwargs)

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
    full_labels = []
    for labels in labels_list:
        full_labels = full_labels + labels
    full_data = data_list[0]
    for other_data in data_list[1:]:
        for key in full_data.keys():
            full_data[key] = np.append(full_data[key],other_data[key],axis=0)
    return full_data,full_labels
