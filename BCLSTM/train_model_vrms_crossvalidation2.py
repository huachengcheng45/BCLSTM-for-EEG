#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:15:00 2022

Cross-validation on the four models
"""

import random
import tensorflow as tf
from tensorflow import keras
from keras import optimizers, losses
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import KFold
from dataset_VRMS_crossvalidation import load_data, preprocess
from sklearn.metrics import mean_squared_error, r2_score
from EEGModels_regression import BCCNN, BCLSTM, BC_CNN_LSTM, BC_CNN_LSTM2


# 设置GPU使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        #打印异常
        print(e)
        
batchsz =16

#Load the dataset
d_all, t_all = load_data(mode='epoch')

n=len(d_all)
A = np.linspace(0,n-1,n,dtype=int)
random.shuffle(A)    

d_all=d_all[A]
t_all=t_all[A,3]

d_all=np.transpose(d_all,(0,2,1,3))

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

historys,test_pred,test_real,mse_test,r2_test=list(),list(),list(),list(),list()#create empty list for saving results
loss_train=list()
loss_val=list()
ind_fold=0

for train_ind,test_ind in kfold.split(d_all,t_all):

    ind_fold=ind_fold+1
    print('fold hao:',ind_fold)
    
    n=len(train_ind)
    A = np.linspace(0,n-1,n,dtype=int)
    random.shuffle(A)    

    epoch_train = d_all[train_ind[A[:int(0.8*n)]]]
    epoch_val = d_all[train_ind[A[int(0.8*n):]]]
    epoch_test = d_all[test_ind]
    label_train=t_all[train_ind[A[:int(0.8*n)]]]
    label_val = t_all[train_ind[A[int(0.8*n):]]]
    label_test=t_all[test_ind]
  
    db_train = tf.data.Dataset.from_tensor_slices((epoch_train,label_train))
    db_val = tf.data.Dataset.from_tensor_slices((epoch_val,label_val))
    db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
    db_val = db_val.shuffle(1000).map(preprocess).batch(batchsz)
    
    model=BCLSTM()
    # model=BCCNN()
    # model=BC_CNN_LSTM()
    # model=BC_CNN_LSTM2()
    
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10
    )
    reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=5)
    
    nan_stopping=keras.callbacks.TerminateOnNaN()

    model.compile(optimizer=optimizers.Nadam(learning_rate=1e-4,clipvalue=1.0,global_clipnorm=1.0),
                    loss=losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
                    metrics=['accuracy'])
    
    print('xunliankaishi!!')
    history  = model.fit(db_train, validation_data=db_val, validation_freq=1,  
                          shuffle=True, epochs=100, callbacks=[early_stopping,reduce_lr,nan_stopping])
    
    history = history.history
    pred_test = model.predict(epoch_test)
    historys.append(history)
    test_pred.append(pred_test)
    test_real.append(label_test)
    
    msefold=mean_squared_error(pred_test,label_test)*85.84*85.84
    r2fold=r2_score(pred_test,label_test)  
    mse_test.append(msefold)
    r2_test.append(r2fold)
    print(f"$$ jun fang wu cha MSE:{msefold}")
    print(f"$$ ni he you du R2:{r2fold}")
    
    loss_val.append(history['val_loss'])
    loss_train.append(history['loss'])
    
    np.save('vrms_BCLSTM_testreal.npy',np.array(test_real,dtype=object))   
    np.save('vrms_BCLSTM_testpred.npy',np.array(test_pred,dtype=object))   
    np.save('vrms_BCLSTM_mse.npy',np.array(mse_test,dtype=object))   
    np.save('vrms_BCLSTM_r2.npy',np.array(r2_test,dtype=object))   
    np.save('vrms_BCLSTM_losstrain.npy',loss_train)   
    np.save('vrms_BCLSTM_lossval.npy',loss_val)  
    model.save('vrms_BCLSTM_model.h5')


mmse=np.mean(mse_test)
smse=np.std(mse_test)
mr2=np.mean(r2_test)
sr2=np.std(r2_test)
print(f"mean and std. mse  is:{mmse}±{smse}")
print(f"mean and std. r2  is:{mr2}±{sr2}")

