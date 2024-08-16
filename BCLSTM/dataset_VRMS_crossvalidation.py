#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:45:10 2022

@author: hccwz
"""

import random
import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
from sklearn.preprocessing import normalize

def load_data(mode='epoch'):
        
    rest0=sio.loadmat(r'E:/study/SCI13/rest0_epoch4class')
    rest=sio.loadmat(r'E:/study/SCI13/rest_epoch4class')
    task=sio.loadmat(r'E:/study/SCI13/task_epoch4class')
    n=2724
    A = np.linspace(0,n-1,n,dtype=int)
    random.shuffle(A) 
    rest1=rest0['rest1_e'][A,:,:]
    n=2730
    A = np.linspace(0,n-1,n,dtype=int)
    random.shuffle(A) 
    rest2=rest0['rest2_e'][A,:,:]
    data_all=np.concatenate((rest1[:500,:,:],rest2[:500,:,:],rest['rest_e'],task['task_e']))

    for i in range(7335):
        data_all[i,:,:]=normalize(data_all[i,:,:],axis=1,norm='l1')

    data_all=np.expand_dims(data_all,axis=3)

  
    label_all=np.concatenate((np.zeros((500,4)),np.zeros((500,4)),rest['rest_ssq'],task['task_ssq']))
    label_all[:,0]=label_all[:,0]/222.72
    label_all[:,1]=label_all[:,1]/114.48
    label_all[:,2]=label_all[:,2]/106.12
    label_all[:,3]=label_all[:,3]/85.84

    n=len(label_all)
    A = np.linspace(0,n-1,n,dtype=int)
    random.shuffle(A)    
    
    data_all=data_all[A]
    label_all=label_all[A]

    return data_all,label_all


def preprocess(x,y):

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    return x, y

