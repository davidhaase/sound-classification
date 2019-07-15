from pipeline import *
from model_generator import get_3_log_model

model = get_3_log_model()

paths = get_all_paths("/Users/noah/Flatiron/NN/AudioRecognition/testaudio/pickles/processed-audio/")

data_list,labels_list = list_of_pickle_paths_to_data_list_and_labels(paths)
data,labels = concate_multiprocess_outputs(data_list,labels_list)



resuts = fit_dict(data,labels,model,["log_S","log_Sh","log_Sp"],epochs=1)
print(results)
