import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv3D,MaxPool3D, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# Defining Inputs and Model Layers
def GetClinicalCNN():

    inputs_feature = keras.Input(shape=(64,64,64,1),name='Image-Input')
    input_clinical = keras.Input(shape=(3),name='Feature-Input')
    inputs = [inputs_feature,input_clinical]

    conv_1       =   Conv3D(filters=32,kernel_size=(5,5,5),strides=(1,1,1),padding='valid',activation='linear',
                        kernel_regularizer=l2(0.9999e-05),name='FirstConvLayer')
    BN_1         =   BatchNormalization(name='BNFirstLayer')
    leakyrelu_1  =   LeakyReLU(alpha=0.1,name='Leaky1stLayer')
    maxpool_1    =   MaxPool3D(pool_size=(3,3,3),strides=(3,3,3),padding='valid',name='MaxPoolFirstLayer')
    drop_1       =   Dropout(0.25,name='DropOutFirstLayer')                     

    conv_2       =   Conv3D(filters=64,kernel_size=(3,3,3),strides=(1,1,1),padding='valid',activation='linear',
                        kernel_regularizer=l2(0.9999e-05),name='SecondConvLayer')
    BN_2         =   BatchNormalization(name='BNSecondLayer')
    leakyrelu_2  =   LeakyReLU(alpha=0.1,name='Leaky2ndLayer')
    maxpool_2    =   MaxPool3D(pool_size=(3,3,3),strides=(3,3,3),padding='valid',name='MaxPoolSecondLayer')
    drop_2       =   Dropout(0.25,name='DropOutSecondLayer')

    conv_3       =   Conv3D(filters=128,kernel_size=(3,3,3),strides=(1,1,1),padding='valid',activation='linear',
                        kernel_regularizer=l2(0.9999e-05),name='ThirdConvLayer')
    BN_3         =   BatchNormalization(name='BNThirdLayer')
    leakyrelu_3  =   LeakyReLU(alpha=0.1,name='Leaky3rdLayer')
    maxpool_3    =   MaxPool3D(pool_size=(3,3,3),strides=(3,3,3),padding='valid',name='MaxPoolThirdLayer')
    drop_3       =   Dropout(0.25,name='DropOutThirdLayer')

    conv_4       =   Conv3D(filters=256,kernel_size=(3,3,3),strides=(1,1,1),padding='valid',activation='linear',
                        kernel_regularizer=l2(0.9999e-05),name='FourthConvLayer')
    BN_4         =   BatchNormalization(name='BNFourthLayer')
    leakyrelu_4  =   LeakyReLU(alpha=0.1,name='Leaky4thLayer')
    maxpool_4    =   MaxPool3D(pool_size=(3,3,3),strides=(3,3,3),padding='valid',name='MaxPoolFourthLayer')
    drop_4       =   Dropout(0.25,name='DropOutFourthLayer')


    flatten = Flatten()
    dense1 = Dense(256,activation='linear',kernel_regularizer=l2(0.9999e-05),name='FirstDenseLayer')
    BN_Dense1    =   BatchNormalization(name='BNDense1Layer')
    leakyrelu_D1  =   LeakyReLU(alpha=0.1,name='LeakyD1Layer')
    drop_d1       =   Dropout(0.25,name='DropOutD1Layer')

    dense2 = Dense(128,activation='linear',kernel_regularizer=l2(0.9999e-05),name='SecondDenseLayer')
    BN_Dense2    =   BatchNormalization(name='BNDense2Layer')
    leakyrelu_D2  =   LeakyReLU(alpha=0.1,name='LeakyD2Layer')
    drop_d2       =   Dropout(0.25,name='DropOutD2Layer')

    dense3 = Dense(2,activation='linear',kernel_regularizer=l2(0.9999e-05),name='ThirdDenseLayer')
    BN_Dense3    =   BatchNormalization(name='BNDense3Layer')
    leakyrelu_D3  =   LeakyReLU(alpha=0.1,name='LeakyD3Layer')
    drop_d3       =   Dropout(0.20,name='DropOutD3Layer')

    dense4 = Dense(1,activation='linear',kernel_regularizer=l2(0.9999e-05),name='OutputLayer')
    BN_Dense4    =   BatchNormalization(name='BNDense4Layer')
    leakyrelu_D4  =   LeakyReLU(alpha=0.1,name='LeakyD4Layer')
    activation = tf.keras.layers.Activation('sigmoid')

    # Defining Model Architecture

    x = conv_1(inputs[0])
    x = BN_1(x)
    #x = maxpool_1(x)
    #x = drop_1(x)

    x = conv_2(x)
    x = BN_2(x)
    x = maxpool_2(x)
    x = drop_2(x)

    x = conv_3(x)
    x = BN_3(x)
    #x = maxpool_3(x)
    x = drop_3(x)


    x = conv_4(x)
    x = BN_4(x)
    x = maxpool_4(x)
    x = drop_4(x)


    x = flatten(x)

    x = dense1(x)
    x = BN_Dense1(x)
    x = leakyrelu_D1(x)
    x = drop_d1(x)

    x = dense2(x)
    x = BN_Dense2(x)
    x = leakyrelu_D2(x)
    x = drop_d2(x)

    x = dense3(x)
    x = tf.concat([x,inputs[1]], 1)
    x = BN_Dense3(x)
    x = leakyrelu_D3(x)
    x = drop_d3(x)



    x = dense4(x)
    x = BN_Dense4(x)

    output = activation(x)

    model = Model(inputs=inputs, outputs=output, name='Functionalmodel')

    return model