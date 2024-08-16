"""
There are four models
BCCNN (Brain Connection based on CNN)
If you feel that the source code of this model is helpful for your study, please cite the reference in your publication.
[1] Chengcheng H , Hong W , Jichi C ,et al.Novel functional brain network methods based on CNN with an application in proficiency evaluation[J].Neurocomputing, 2019, 359:153-162.DOI:10.1016/j.neucom.2019.05.088.

BCLSTM (Brain Connection based on LSTM)
BC-CNNLSTM 
BC-CNN-LSTM2 (Combination of BCCNN and BCLSTM)

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers,regularizers
from tensorflow.keras.layers import Dense, Dropout, Lambda, Input, ReLU, Flatten, Conv2D, MaxPooling2D, GRU,LSTM, BatchNormalization, Concatenate

    
#@tf.function 
# Compute the Pearson correlation among the electrodes    
def LinearCorr(inputs):
    size=inputs.shape
   # print('size:',size)
    N=size[2]
    xy_mut = tf.matmul(tf.transpose(inputs,perm=[0,3,1,2]), tf.transpose(inputs,perm=[0,3,2,1]))
    #print(xy_mut.shape)
    x_sqr=tf.square(inputs)
    x_sqr_add=tf.reduce_sum(x_sqr,2,keepdims=True)
    x_add=tf.reduce_sum(inputs,2,keepdims=True)
    #print(x_sqr_add.shape,x_add.shape)
    rep=tf.sqrt(N*x_sqr_add-x_add**2)
   # print(rep.shape)
    corr=(N*xy_mut-tf.matmul(
        tf.transpose(x_add,perm=[0,3,1,2]),tf.transpose(
            x_add,perm=[0,3,2,1])))/tf.matmul(
                tf.transpose(rep,perm=[0,3,1,2]), tf.transpose(rep,perm=[0,3,2,1]))
   # print(corr.shape)
    return tf.transpose(corr,perm=[0,2,3,1])


def BCCNN(nb_classes=1, Chans=30, Samples=200, filter_length=25,dropoutRate=0.3):
    
    inp_all=layers.Input(shape=(Chans,Samples,1))
    
    # Input and CNN
    x=layers.BatchNormalization()(inp_all)
    x=Conv2D(20,kernel_size=(1,10),strides=(1,1),
                  activation=keras.activations.linear,input_shape=(Chans,Samples,1), name='conv_1d'
                  )(x)
    
    # Compute the Pearson correlation
    AM=Lambda(LinearCorr)(x)

    # Extract the functiaonl brain network features with CNN blocks
    bn1=BatchNormalization()(AM)
    bn1=Conv2D(16,kernel_size=2,strides=1,name='conv2',activation=tf.nn.relu,
                kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01))(bn1)# 第一个卷积层, 6 个 3x3 卷积核
    bn1=MaxPooling2D(pool_size=2,strides=2)(bn1) # 高宽各减半的池化层
    
    bn2=BatchNormalization()(bn1)
    bn2=Conv2D(32,kernel_size=2,strides=1,name='conv3',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn2) # 第二个卷积层, 16 个 3x3 卷积核
    bn2=MaxPooling2D(pool_size=2,strides=2)(bn2) # 高宽各减半的池化层
    
    bn3=BatchNormalization()(bn2)
    bn3=Conv2D(64,kernel_size=2,strides=1,name='conv4',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn3) # 第二个卷积层, 16 个 3x3 卷积核
    bn3=MaxPooling2D(pool_size=2,strides=2)(bn3) # 高宽各减半的池化层
    
    # Regression
    output=Flatten()(bn3)# 打平层，方便全连接层处理
    output=Dropout(dropoutRate)(output)
    output=Dense(128, activation='relu')(output)
    layers.Dropout(dropoutRate)(output)
    output=Dense(64, activation='relu')(output)
    output=Dense(32, activation='relu')(output)
    softmax=Dense(1, activation='sigmoid')(output)

    return Model(inputs=inp_all, outputs=softmax)   


def BCLSTM(nb_classes=1, Chans=30, Samples=200, dropoutRate=0.3):
    
    inp_all=layers.Input(shape=(Chans,Samples,1))
    
    #LSTM model       
    inp=layers.Input(shape=(Samples,1)) 
    x=LSTM(20,return_sequences=True)(inp)
    model=Model(inputs=inp,outputs=x)
        
    # Input and split
    x=layers.BatchNormalization()(inp_all)
    chan_split=tf.unstack(x,axis=1)
    
    # Siamses LSTM
    models=[]
    for ch in range(30):
        models.append(tf.expand_dims(model(chan_split[ch]),1))
    block1=Concatenate(axis=1)(models)

    # Compute the Pearson correlation
    block2=Lambda(LinearCorr)(block1)

    # Extract the functiaonl brain network features with CNN blocks
    bn1=BatchNormalization()(block2)
    bn1=Conv2D(16,kernel_size=2,strides=1,name='conv2',activation=tf.nn.relu,
                kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01))(bn1)# 第一个卷积层, 6 个 3x3 卷积核
    bn1=MaxPooling2D(pool_size=2,strides=2)(bn1) # 高宽各减半的池化层
    
    bn2=BatchNormalization()(bn1)
    bn2=Conv2D(32,kernel_size=2,strides=1,name='conv3',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn2) # 第二个卷积层, 16 个 3x3 卷积核
    bn2=MaxPooling2D(pool_size=2,strides=2)(bn2) # 高宽各减半的池化层
    
    bn3=BatchNormalization()(bn2)
    bn3=Conv2D(64,kernel_size=2,strides=1,name='conv4',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn3) # 第二个卷积层, 16 个 3x3 卷积核
    bn3=MaxPooling2D(pool_size=2,strides=2)(bn3) # 高宽各减半的池化层
    
    # Regression
    output=Flatten()(bn3)# 打平层，方便全连接层处理
    output=Dropout(dropoutRate)(output)
    output=Dense(128, activation='relu')(output)
    layers.Dropout(dropoutRate)(output)
    output=Dense(64, activation='relu')(output)
    output=Dense(32, activation='relu')(output)
    softmax=Dense(1, activation='sigmoid')(output)

    return Model(inputs=inp_all, outputs=softmax)   


def BC_CNN_LSTM(nb_classes=1, Chans=30, Samples=200, dropoutRate=0.3):
    
    inp_all=layers.Input(shape=(Chans,Samples,1))
    
    #LSTM model       
    inp=layers.Input(shape=(Samples,1)) 
    x=LSTM(20,return_sequences=True)(inp)
    model=Model(inputs=inp,outputs=x)
        
    # Input, CNN and split
    x=layers.BatchNormalization()(inp_all)
    x=Conv2D(20,kernel_size=(1,10),strides=(1,1),
                  activation=keras.activations.linear,input_shape=(Chans,Samples,1), name='conv_1d'
                  )(x)
    chan_split=tf.unstack(x,axis=1)
    
    # Siamses LSTM
    models=[]
    for ch in range(30):
        models.append(tf.expand_dims(model(chan_split[ch]),1))
    block1=Concatenate(axis=1)(models)

    # Compute the Pearson correlation
    AM=Lambda(LinearCorr)(block1)

    # Extract the functiaonl brain network features with CNN blocks
    bn1=BatchNormalization()(AM)
    bn1=Conv2D(16,kernel_size=2,strides=1,name='conv2',activation=tf.nn.relu,
                kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01))(bn1)# 第一个卷积层, 6 个 3x3 卷积核
    bn1=MaxPooling2D(pool_size=2,strides=2)(bn1) # 高宽各减半的池化层
    
    bn2=BatchNormalization()(bn1)
    bn2=Conv2D(32,kernel_size=2,strides=1,name='conv3',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn2) # 第二个卷积层, 16 个 3x3 卷积核
    bn2=MaxPooling2D(pool_size=2,strides=2)(bn2) # 高宽各减半的池化层
    
    bn3=BatchNormalization()(bn2)
    bn3=Conv2D(64,kernel_size=2,strides=1,name='conv4',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn3) # 第二个卷积层, 16 个 3x3 卷积核
    bn3=MaxPooling2D(pool_size=2,strides=2)(bn3) # 高宽各减半的池化层
    
    # Regression
    output=Flatten()(bn3)# 打平层，方便全连接层处理
    output=Dropout(dropoutRate)(output)
    output=Dense(128, activation='relu')(output)
    layers.Dropout(dropoutRate)(output)
    output=Dense(64, activation='relu')(output)
    output=Dense(32, activation='relu')(output)
    softmax=Dense(1, activation='sigmoid')(output)

    return Model(inputs=inp_all, outputs=softmax)   


def BC_CNN_LSTM2(nb_classes=1, Chans=30, Samples=200, dropoutRate=0.3):
    
    inp=layers.Input(shape=(Samples,1)) 
        
    #LSTM model      
    x=LSTM(20,return_sequences=True)(inp)
    model=Model(inputs=inp,outputs=x)
    inp_all=layers.Input(shape=(Chans,Samples,1))
    
    # Input, CNN and split
    x=layers.BatchNormalization()(inp_all)
    x=Conv2D(20,kernel_size=(1,10),strides=(1,1),
                  activation=keras.activations.linear,input_shape=(Chans,Samples,1), name='conv_1d'
                  )(x)
    chan_split=tf.unstack(x,axis=1)
    
    # Siamses LSTM
    models=[]
    for ch in range(30):
        models.append(tf.expand_dims(model(chan_split[ch]),1))
    block1=Concatenate(axis=1)(models)

    # Compute the Pearson correlation
    block2=Lambda(LinearCorr)(block1)
    block3=Lambda(LinearCorr)(x)
    AMs=Concatenate(axis=-1)([block2,block3])

    # Extract the functiaonl brain network features with CNN blocks
    bn1=BatchNormalization()(AMs)
    bn1=Conv2D(16,kernel_size=2,strides=1,name='conv2',activation=tf.nn.relu,
                kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01))(bn1)# 第一个卷积层, 6 个 3x3 卷积核
    bn1=MaxPooling2D(pool_size=2,strides=2)(bn1) # 高宽各减半的池化层
    
    bn2=BatchNormalization()(bn1)
    bn2=Conv2D(32,kernel_size=2,strides=1,name='conv3',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn2) # 第二个卷积层, 16 个 3x3 卷积核
    bn2=MaxPooling2D(pool_size=2,strides=2)(bn2) # 高宽各减半的池化层
    
    bn3=BatchNormalization()(bn2)
    bn3=Conv2D(64,kernel_size=2,strides=1,name='conv4',activation=tf.nn.relu,
               kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l2(0.01))(bn3) # 第二个卷积层, 16 个 3x3 卷积核
    bn3=MaxPooling2D(pool_size=2,strides=2)(bn3) # 高宽各减半的池化层
    
    # Regression
    output=Flatten()(bn3)# 打平层，方便全连接层处理
    output=Dropout(dropoutRate)(output)
    output=Dense(128, activation='relu')(output)
    layers.Dropout(dropoutRate)(output)
    output=Dense(64, activation='relu')(output)
    output=Dense(32, activation='relu')(output)
    softmax=Dense(1, activation='sigmoid')(output)

    return Model(inputs=inp_all, outputs=softmax)  

