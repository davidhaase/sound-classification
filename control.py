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
train_file = '/Volumes/LaCie Rose Gold 2TB/Datasets/Features/feat_003/Train_Feature_Pickles/extracted.pkl'
test_file = '/Volumes/LaCie Rose Gold 2TB/Datasets/Features/feat_006/Test_Feature_Pickles/extracted.pkl'
prediction_csv = '/Volumes/LaCie Rose Gold 2TB/Datasets/Features/feat_006/predictions.csv'

train_df = pd.read_pickle(train_file)
test_df = pd.read_pickle(test_file)

test_size = 0.2
random_state = 23
num_epochs = 100

target = train_df['Language']
features = train_df.drop('Language', axis=1)
X_train, val_X, y_train, val_y = train_test_split(features, target, test_size=test_size, random_state=random_state)

X_train = np.array(X_train.Features.tolist())
y_train = np.array(y_train.tolist())

val_X = np.array(val_X.Features.tolist())
val_y = np.array(val_y.tolist())

lb = LabelEncoder()


y_train = np_utils.to_categorical(lb.fit_transform(y_train))
val_y = np_utils.to_categorical(lb.fit_transform(val_y))

pickle.dump(lb, open('encoder.pkl','wb+'))

num_labels = y_train.shape[1]
filter_size = 2
model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(val_X, val_y), shuffle=False, verbose=1)
model.save('my_model.h5')
# new_model = load_model('my_model.h5')

X_predict = np.array(test_df.Features.tolist())
predictions = new_model.predict_classes(X_predict)
test_df['prediction_number'] = predictions
test_df['prediction_label'] = lb.inverse_transform(predictions)
test_df.drop(['File_ID', 'prediction_number','Path', 'Features', 'Lang_ID'], axis=1, inplace=True)

test_df['Correct'] = False
test_df.loc[test_df['Language'] == test_df['prediction_label'], 'Correct'] = True
correct = test_df['Correct'].sum()
total = len(test_df)
print('{} correct out of {}: {}%'.format(correct, total, correct/total))
test_df.to_csv(prediction_csv, index=False)
# plt.figure(figsize=(18, 10))
# acc = list(history.history['acc'])
# val_acc = list(history.history['val_acc'])
# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model_accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.savefig('output/accuracy.png')
# plt.show()
#
# plt.figure(figsize=(18, 10))
# loss = list(history.history['loss'])
# val_loss = list(history.history['val_loss'])
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model_loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.savefig('output/loss.png')
# plt.show()
