#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:45:24 2017

@author: caradumelvin
"""
from Preprocessing import Get_Data
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne 
import time
import pandas as pd
from tqdm import tqdm
import h5py 

'''
Dict_path = '/melvin/Data/Quora/glove.6B.50d.txt'
Data_path = '/melvin/Data/Quora/'
Test_Data_path = '/melvin/Data/Quora/'
'''
Data_path = '/Users/caradumelvin/Documents/Machine Learning - IA/Natural Language Processing/Quora/'
Test_Data_path = Data_path
Dict_path = '/Users/caradumelvin/Documents/Machine Learning - IA/Natural Language Processing/glove.6B/glove.6B.50d.txt'


def batch_gen(X, y, Mask, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('float32'), Mask[idx].astype('float32')

   
def GRU(N_HIDDEN1):
    print("Building Gated Recurrent Network")
    l_in = lasagne.layers.InputLayer(shape=(None, None, 1))
    l_mask = lasagne.layers.InputLayer(shape=(None, None))
    
    gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
                                                    W_hid=lasagne.init.Orthogonal(),
                                                    b=lasagne.init.Constant(0.))
    
    l_GRU1 = lasagne.layers.recurrent.GRULayer(l_in, N_HIDDEN1, resetgate=gate_parameters,
                                                updategate=gate_parameters, mask_input=l_mask,
                                                learn_init = True, grad_clipping=100, backwards=True,
                                                hidden_update=gate_parameters)
    
    l_GRU1_back = lasagne.layers.recurrent.GRULayer(l_in, N_HIDDEN1, resetgate=gate_parameters,
                                                updategate=gate_parameters, mask_input=l_mask, 
                                                learn_init=True, grad_clipping=100, backwards=True,
                                                hidden_update=gate_parameters)
    
    l_sum = lasagne.layers.ElemwiseSumLayer([l_GRU1, l_GRU1_back])
    
    
    l_lstm_slice = lasagne.layers.SliceLayer(l_sum, 0, 1)
    
    l_out = lasagne.layers.DenseLayer(l_lstm_slice, num_units=2, 
                                      nonlinearity=lasagne.nonlinearities.sigmoid)
    
    return l_in, l_mask, l_out

def Train_model(BATCH_SIZE, number_of_epochs, lr):
    try:
        X_train, X_val, X_test, y_train, y_val, mask_train, mask_val, mask_test = Get_Data(as_list=False)
        print("h5 Dataset created in current directory")
    except:
        print("Loading existing Dataset")
        f = h5py.File("Quora_Data", "r")
        grp = f[u'X']
        X_train, X_test, X_val = grp["X_train"], grp["X_test"], grp["X_val"]
        y_train, y_val = grp["y_train"], grp["y_val"]
        mask_train, mask_val, mask_test = grp["mask_train"], grp["mask_test"], grp["mask_val"]
    
    l_in, l_mask, l_out = GRU(N_HIDDEN1=16)
    
    y_sym = T.matrix()

    output = lasagne.layers.get_output(l_out)
    pred = (output > 0.5)

    loss = T.mean(lasagne.objectives.binary_crossentropy(output, y_sym))

    acc = T.mean(T.eq(pred, y_sym))

    params = lasagne.layers.get_all_params(l_out)
    grad = T.grad(loss, params)
    updates = lasagne.updates.rmsprop(grad, params, learning_rate=lr)

    f_train = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc], updates=updates)
    f_val = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc])
    f_predict_probas = theano.function([l_in.input_var, l_mask.input_var], output)
    
    N_BATCHES = len(X_train) // BATCH_SIZE
    N_VAL_BATCHES = len(X_val) // BATCH_SIZE
    train_batches = batch_gen(X_train[:], y_train[:], mask_train[:], BATCH_SIZE)
    val_batches = batch_gen(X_val[:], y_val[:], mask_val[:], BATCH_SIZE)

    print("Start training")
    for epoch in range(number_of_epochs):
        train_loss = 0
        train_acc = 0
        start_time = time.time()
        for _ in range(N_BATCHES):
            X, y, mask = next(train_batches)
            loss, acc = f_train(X, y, mask)
            train_loss += loss
            train_acc += acc
        train_loss /= N_BATCHES
        train_acc /= N_BATCHES
            
        val_loss = 0
        val_acc = 0
        for _ in range(N_VAL_BATCHES):
            X, y, mask = next(val_batches)
            loss, acc = f_val(X, y, mask)
            val_loss += loss
            val_acc += acc
        val_loss /= N_VAL_BATCHES
        val_acc /= N_VAL_BATCHES
        print("Epoch {} of {} took {:.3f}s".format(
              epoch + 1, number_of_epochs, time.time() - start_time))
        print('  Train loss: {:.03f} - Validation Loss: {:.03f}'.format(
              train_loss, val_loss))
        print('  Train accuracy: {:.03f}'.format(train_acc))
        print('  Validation accuracy: {:.03f}'.format(val_acc))
    '''
    print("Testing model: ")
    predictions = np.zeros((len(X_test), 2))
    for i in tqdm(range(len(X_test))):
        predictions[i] = f_predict_probas(X_test[i:i+1], mask_test[i:i+1])
    predictions = predictions[:,0]
    print("Test set predicted successfully, good luck !")
    
    Submission = pd.DataFrame({'id': [i for i in range(1, len(predictions)+1)],
                                      'label': predictions})
    
    Submission.to_csv(os.path.join(Data_path,r'Submission.csv'), header=True, index=False)        
    print("CSV submission file generated on working directory")
    np.savez('model_lin.npz', *lasagne.layers.get_all_param_values(l_in))
    np.savez('model_lmask.npz', *lasagne.layers.get_all_param_values(l_mask))
    np.savez('model_lout.npz', *lasagne.layers.get_all_param_values(l_out))
    '''

if __name__ == "__main__":
    #Don't forget to change paths before AWS script !!!!!!
    Train_model(BATCH_SIZE=128, number_of_epochs=5, lr=0.005)
