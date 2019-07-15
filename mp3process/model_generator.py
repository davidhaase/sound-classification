from keras import Model,Input
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,concatenate
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
    outputs = Dense(100,activation="relu",use_bias=True)(x)
    return Model(inputs=input_layer,outputs=outputs)

def get_3_log_model():
    inputs = [Input((128,433,1)) for i in range(3)]
    mods = [log_attach(inputs[i]) for i in range(3)]

    com = concatenate([mod.output for mod in mods])
    x = Dense(100,activation="relu",use_bias=True)(com)
    x = Dense(50,activation="relu",use_bias=True)(x)
    x = Dense(5,activation="softmax",use_bias=True)(x)
    mod = Model(inputs = inputs,outputs = x)
    opt = Adam(lr=.01)
    mod.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])
    return mod
