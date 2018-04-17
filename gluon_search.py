from __future__ import print_function
import time
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import random

def NN_build(num_hidden, activations):
    """A function for building container neural network 
    for MXNet. This is specifically designed for NNs with an output layer size 2 (typical classification algos).
    Extension for multiclass may be added in the future

    num_hidden: hidden layers, of list form, length will be number of layers, values should be size of that layer
    activations: list, length number of hidden layers, values string for activation function"""

    num_outputs = 2
    net = gluon.nn.Sequential()
    with net.name_scope():
        for i in np.arange(len(num_hidden)):
            net.add(gluon.nn.Dense(num_hidden[i], activation = activations[i]))
            net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(num_outputs, activation = 'relu'))
    return(net)

def softmax(y):
    import math
    out = np.zeros((y.shape[0], y.shape[1]))
    for i in np.arange(y.shape[0]):
        exps = [math.exp(k) for k in y[i]]
        softmax = [k/sum(exps) for k in exps]
        out[i] = softmax
    return(out)

def search_over_NNs(X_tr, X_test, Y_tr, Y_test, epochs, num_hidden_arr, activations_arr, loss_func, init_std = 0.1, learning_rate =.01, momentum = 0.9):
    """Function for searching over Gluon NN architectures and hyperparameters. 
    X_tr: original training data, numpy array
    X_test: original test data, numpy array
    Y_tr: numpy array of training labels
    Y_test: numpy array of test labels
    epochs: number of iterations over training data
    num_hidden_arr: array of num_hidden lists, providing the different architectures
    acitvations_arr: array of activations lists, giving the activation functions of the layers
    loss_func: a loss function from Gluon loss API 
    init_std: standard deviation of weight initialisation (fixed to be normal mean 0)
    learning_rate: the learning rate!
    momentum: the momentum!"""
    from sklearn.metrics import log_loss, accuracy_score
    import pandas as pd

    mx_X_tr = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(np.array(X_tr), np.array(Y_tr).astype(int)), batch_size = 64, shuffle = True)
    mx_X_test = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(np.array(X_test), np.array(Y_test).astype(int)), batch_size = 64, shuffle = True)
    import time
   
    for j in np.arange(num_hidden_arr.shape[0]):
        print('-----------------------------------------------\n')
        print('Building model with hidden layers :', num_hidden_arr[j])
        net = NN_build(num_hidden_arr[j], activations_arr[j])
        
        model_ctx = mx.cpu()
        net.collect_params().initialize(mx.init.Normal(sigma=init_std), ctx = model_ctx)
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'momentum': momentum})
        metrics = {}
        metrics['train_losses'] = np.zeros(epochs)
        metrics['time'] = np.zeros(epochs)
        metrics['train_accuracy'] = np.zeros(epochs)
        metrics['test_losses'] = np.zeros(epochs)
        metrics['test_accuracy'] np.zeros(epochs)
        metrics['model_id'] = random.randint(3,100)*np.ones(epochs)
        model_dict = {'model_id': int(metrics['model_id'][0]), 'hidden_layers': list(num_hidden_arr[j]), 'activations': 
                     list(activations_arr[j])}
        
        
        for e in range(epochs):
            start = time.time()
            y_tr_pred = np.array([])
            for i, (data, label) in enumerate(mx_X_tr):
                data = data.as_in_context(mx.cpu()).reshape((-1,X_tr.shape[1])).astype('float32')
                label = label.as_in_context(mx.cpu()).astype('float32')
                with autograd.record(): # Start recording the derivatives
                    output = net(data) # the forward iteration
                    loss = loss_func(output, label)
                loss.backward()
                trainer.step(data.shape[0])
                # Provide stats on the improvement of the model over each epoch
                curr_loss = mx.ndarray.mean(loss).asscalar()
                y_tr_pred = np.append(y_tr_pred, output.asnumpy()) 
            

            y_tr_pred = y_tr_pred.reshape((X_tr.shape[0], 2))
            y_tr_pred = softmax(y_tr_pred)[:,1]
            y_tr_pred_labels = np.where(y_tr_pred > 0.5, 1, 0)
            metrics['train_losses'][e] = log_loss(np.array(1*Y_tr).astype('int32'), y_tr_pred) 
            metrics['train_accuracy'][e] = accuracy_score(np.array(1*Y_tr).astype('int32'), y_tr_pred_labels)
            metrics['time'][e] = time.time()-start

            y_pred = np.array([])
            for (data,label) in mx_X_test:
                data = data.as_in_context(mx.cpu()).astype('float32')
                label = label.as_in_context(mx.cpu()).astype('float32')
                output = net(data)
                y_pred = np.append(y_pred, output.asnumpy()) 
            
            y_pred = y_pred.reshape((X_test.shape[0], 2))
            y_pred = softmax(y_pred)[:,1]
            y_pred_labels = np.where(y_pred > 0.5, 1, 0)
            metrics['test_losses'][e] = log_loss(np.array(1*Y_test).astype('int32'), y_pred) 
            metrics['test_accuracy'][e] = accuracy_score(np.array(1*Y_test).astype('int32'), y_pred_labels)
            
            if (e+1) % 10 == 0:
                print("Epoch {} Train acc: {}  Test acc: {}".format(str(e+1).ljust(3), 
                                                                    np.round(metrics['train_accuracy'][e],3),
                                                                    np.round(metrics['test_accuracy'][e],3)))
        if j == 0:
            all_metrics = pd.DataFrame(metrics).reset_index()
            all_metrics['index'] = 1 + np.arange(epochs)
        else:
            metrics = pd.DataFrame(metrics).reset_index()
            metrics['index'] = 1 + np.arange(epochs)
            all_metrics = all_metrics.append(pd.DataFrame(metrics))
            
        
    all_metrics.rename(columns = {'index':'epoch'}, inplace = True)
        
    return(all_metrics, net)
