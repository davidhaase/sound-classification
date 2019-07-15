from keras import Model,Input
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,concatenate,Dropout
from keras.regularizers import l2
from keras.optimizers import Adam


def log_attach(input_layer):
    x = Conv2D(128,kernel_size=4,padding="valid",data_format="channels_last",activation="relu",use_bias=True)(input_layer)
    x = MaxPooling2D(padding="valid",pool_size=2)(x)
    x = Conv2D(64,kernel_size=4,padding="valid",data_format="channels_last",activation="relu",use_bias=True)(x)
    x = MaxPooling2D(padding="valid",pool_size=2)(x)
    x = Conv2D(5,kernel_size = 3,padding = "valid",data_format = "channels_last",activation = "relu",use_bias=True)(x)
    x = MaxPooling2D(padding="valid",pool_size=2)(x)
    x = Flatten()(x)
    outputs = Dense(50,activation="relu",use_bias=True)(x)
    return Model(inputs=input_layer,outputs=outputs)


'''
0>num_of_log_transforms>=3
type(num_of_log_transforms) == int
'''
def get_log_model(num_of_log_transforms):
    inputs = [Input((128,433,1)) for i in range(num_of_log_transforms)]
    mods = [log_attach(inputs[i]) for i in range(num_of_log_transforms)]
    skip_first = False
    if num_of_log_transforms == 1:
        skip_first = True
        x = mods[0].output
        inputs = inputs[0]
    else:
        x = concatenate([mod.output for mod in mods])
    if not skip_first:
        x = Dense(100,activation="relu",use_bias=True)(com)

    mod = Model(inputs = inputs,outputs = x)
    opt = Adam(lr=.01)
    mod.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])
    return mod
def get_mfcc_model():
    input_layer = Input((13,433,1))
    x = Conv2D(128,kernel_size=(4,3),padding="valid",data_format="channels_last",activation="relu",use_bias=True)(input_layer)
    x = MaxPooling2D(padding="valid",pool_size=2)(x)
    x = Dropout(.5)(x)
    x = Conv2D(32,kernel_size=3,padding="valid",data_format="channels_last",activation="relu",use_bias=True)(x)
    x = MaxPooling2D(padding="valid",pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(100,activation="relu",use_bias=True)(x)
    x = Dense(50,activation="relu",use_bias=True)(x)
    outputs = Dense(5,activation="softmax",use_bias=True)(x)
    mod = Model(inputs=input_layer,outputs=outputs)
    opt = Adam(lr=.01)
    mod.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])
    return mod
