from librosaprocess import process
import re
import numpy as np
import pickle

def remove_nums(string):
    string = string[string.rfind('/')+1:]
    string = string.replace(".mp3","")
    return re.sub(r'[0-9]+', '', string)
def multiprocess(file_list):
    sample_amount = len(file_list)
    current_index = 0
    first_data = process(file_list[current_index])

    all_data = {}
    labels = []
    for key in first_data.keys():
        data_shape = list(first_data[key].shape)
        new_shape = tuple([sample_amount] + data_shape)
        print(f"{key} shape {new_shape}")
        all_data[key] = np.ndarray(new_shape)

    while current_index < sample_amount:
        file_name = file_list[current_index]
        labels.append(remove_nums(file_name))
        data = process(file_name)
        for key in data.keys():
            try:
                all_data[key][current_index] = data[key]
            except Exception as e:
                print(key)
                raise e
        current_index+=1

    return all_data,labels



def data_label_write(all_data,labels,filename):
    both = {'data':all_data,'labels':labels}
    pickle.dump( both, open( filename, "wb+" ) )
def data_label_load(filename):
    both = pickle.load(open( filename, "rb" ))
    return both["data"],both["labels"]

#'log_S', 'log_Sh', 'log_Sp', 'C', 'mfcc', 'delta_mfcc', 'delta2_mfcc', 'tempo', 'beats', 'C_sync'#
