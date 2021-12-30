# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:12:18 2017

@author: Dean
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.python.keras.backend import dropout # allow completion to work

from DataTypes import TrainMetrics
import pandas as pd
import time
import os
#from ClockworkRNN import CWRNN

from scripts.DataTypes import FeedLoc

from IPython.display import Markdown, display
def printmd(string, color=None):
    if color is None:
        display(Markdown(string))
    else:
        colorstr = "<span style='color:{}'>{}</span>".format(color, string)
        display(Markdown(colorstr))

def SecToHMS(t):
    """Makes a string like ' 2:15:36' to represent some duration, given in seconds. 8 chars"""
    return f"{t//3600.:2.0f}:{(t%3600)//60.:02.0f}:{t%60.:02.0f}"

#==========================================================================
class ValidationCb(tf.keras.callbacks.Callback):
    """
    This callback is used to validate/test the model after each epoch. 
    Note that Keras supports this functionality with validation_data in .fit(), but a downside
    with that for a LSTM is that it starts the validation without building
    state. Hence, I made this custom callback. This builds state before the prediction range - so the validation
    is a better representation of its actual usage. Allows for better-informed
    early stopping.
    """
    def __init__(self):
        pass
    
    
    def setup(self, inData, outTarget, trainMetrics, patience, maxEpochs):
        self.inData = inData # The input data that will be used for validation
        self.outTarget = np.array(outTarget) # The target output. A perfect prediction model would predict these values
        self.targetSize = outTarget.size # Number of points
        self.targetTimeSteps = outTarget.shape[-2]
        self.maxEpochs = maxEpochs
        
        # Determine the thresholds for penalising lack of movement
        # This is to avoid uneventful results that don't do anything
        # Do this by comparing to a smoothed version of the target
        totalDiff = 0
        diffCount = 0
        windowSz = 10
        window = np.ones((windowSz,))/windowSz
        for sample in range(outTarget.shape[0]):
            for outIdx in range(outTarget.shape[-1]):
                smoothed = np.convolve(outTarget[sample,:,outIdx], window, mode='valid')
                totalDiff += np.sum(np.abs(np.diff(smoothed)))
                diffCount += outTarget[sample,:,outIdx].size-1
        avgDiffTarget = totalDiff / diffCount
        self.avgDiffUpper = avgDiffTarget * 0.25 # above this, there's no penalisation
        self.avgDiffLower = avgDiffTarget * 0.1 # Below this, the penalty is a maximum
        
        self.trainMetrics = trainMetrics # Links to the r.trainMetrics

        # Tracking the best result:
        self.bestFitness = 0
        self.bestWeights = []
        self.bestEpoch = 0
        # Early stopping
        self.wait = 0
        self.patience = patience # Epochs before stopping early. Set to 0 to disable early stopping

        self.prevFitness = 0
        self.stopped_epoch = 0

        self.startTime = time.time()
        self.prevPrintTime = time.time()
        self.printCount = 0 # the number of times we've printed
        self.epochTimeHist = [] # a list of the previous ~3 epoch training durations
           
    #==========================================================================
    def on_epoch_end(self, epoch, logs={}):
        self.trainMetrics.curEpoch += 1
        thisEpoch = self.trainMetrics.curEpoch
        if not self.epochTimeHist:
            self.epochTimeHist.append(self.startTime)

        predictY = self.model.predict(self.inData, batch_size=100)
        # Grab only the relevant time steps
        err = predictY[:,-self.targetTimeSteps:,:] - self.outTarget
        
        val_abs_err = np.sum(np.abs(err)) / self.targetSize
        val_sq_err = np.sum(err**2) / self.targetSize
        
               
        fitness = 1/val_sq_err
        # Penalise predictions that don't vary across the time series
        thisDiff = np.mean(np.abs(np.diff(predictY, axis=1)))
        debug = 0
        if debug: print('Dif Score {:5f}, Lower {:5f}, Upper {:5f}'.format(thisDiff, self.avgDiffLower, self.avgDiffUpper), end='')
        penalty = 1. # fitness is multiplied by this penalty
        if thisDiff < self.avgDiffUpper:
            penaltyLower = 0.001
            penaltyUpper = 1
            pos = (thisDiff - self.avgDiffLower) / (self.avgDiffUpper - self.avgDiffLower) # 0 to 1
            penalty = (pos) * (penaltyUpper - penaltyLower) + penaltyLower
            penalty = np.clip(penalty, penaltyLower, penaltyUpper)
            fitness *= penalty
            if debug: print(' scaler = {:5f}'.format(penalty))
        if debug: print('') # new line
        
        logs['fitness'] = fitness # for choosing the best model
        
        # Check if this is the best result
        bestResult = False
        if fitness > self.bestFitness:
            self.bestFitness = fitness
            self.bestWeights = self.model.get_weights()
            self.bestEpoch = thisEpoch
            bestResult = True
        
        # add the epoch duration to the list
        now = time.time()
        self.epochTimeHist.append(now)

        epochsRemaining = self.maxEpochs - thisEpoch
        timePerEpoch = np.mean(np.diff(self.epochTimeHist[-4:]))
        timeRemaining = epochsRemaining * timePerEpoch

        def PrintHeaders():
            # Headers for the text table printed during training
            print(f"Epoch TrainErrSq ValErrSq Fitness Penalty ProcTime Remaining")

        # Epoch printout
        if bestResult or (thisEpoch%10)==0 or now - self.prevPrintTime > 60. \
            or thisEpoch < 5 or thisEpoch == self.maxEpochs:
            if self.printCount%10 == 0:
                PrintHeaders()
            #print(f"Epoch {thisEpoch:2} - TrainErrSq={logs['loss']:6.3f}, ValErrSq={val_sq_err:6.3f}, Fitness={fitness:6.3f}, " +
                 #f"Penalty= {penalty:5.3f} {' - New best! ' if bestResult else ''}")
            print(f"{thisEpoch:5} {logs['loss']:10.3f} {val_sq_err:8.3f} {fitness:7.3f} {penalty:7.3f} " +
                f"{(now - self.epochTimeHist[-2]):7.1f}s {SecToHMS(timeRemaining):>9s}" + 
                f"{' - New best! ' if bestResult else ''}")
            self.printCount += 1
            self.prevPrintTime = now
                
        # Update trainMetrics
        self.trainMetrics.absErrVal.append(val_abs_err)
        self.trainMetrics.lossVal.append(val_sq_err)
        self.trainMetrics.lossTrain.append(logs['loss'])
        self.trainMetrics.absErrTrain.append(logs['mean_absolute_error'])
        self.trainMetrics.fitness.append(fitness)
        
        # Early stopping
        if self.patience != 0:
            if fitness == self.bestFitness:
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience and fitness < self.prevFitness:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print('No validation improvement in {} epochs'.format(self.patience))
                    print('STOPPING TRAINING EARLY AT EPOCH {}'.format(thisEpoch))
        
        self.prevFitness = fitness
        
    
# To allow pickling:
# https://stackoverflow.com/questions/44855603/typeerror-cant-pickle-thread-lock-objects-in-seq2seq
import tensorflow as tf
setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)


#==========================================================================
# Functions to rearrange data to perform 1 fit call
def timeStepsToDim1X(data):
    """ Reshapes the data: copies the time steps as extra rows in the first dim
    """
    (samples, timeSteps, features) = data.shape
    out = np.zeros((samples*timeSteps, 1, features))
    for i in range(timeSteps):
        out[i*samples:(i+1)*samples] = data[:,i:i+1,:]
    return out

def timeStepsToDim1Y(data):
    """ Same as timeStepsToDim1X, but 2nd dimension is flattened"""
    (samples, timeSteps, features) = data.shape
    out = np.zeros((samples*timeSteps, features))
    for i in range(timeSteps):
        out[i*samples:(i+1)*samples] = data[:,i,:]
    return out

def Dim1ToTimeSteps(data, samples):
    """ Inverse of timeStepsToDim1Y"""
    (dim1, features) = data.shape
    timeSteps = int(dim1 / samples)
    out = np.zeros((samples, timeSteps, features))
    for i in range(timeSteps):
        out[:, i, :] = data[i*samples:(i+1)*samples, :]
    return out

#==========================================================================
# SPLIT UP THE DATA
# Training, Validation, Testing, and Exclude
# Exclude fixed number of points at the end. Then split into:if
# Testing, Training, Validation (in that order)
def _CalcIndices(tMax, dataRatios, exclude):
    """Determine the indices for 3 ranges, with ratios defined by dataRatios
    """    
    pos = 0
    tInd = list(range(len(dataRatios)))
    tEnd = tMax - exclude
    for i in range(len(dataRatios)):
        j = (i+2)%len(dataRatios) # Test data, then training, then validation
        tInd[j] = (np.arange(pos, min(tEnd, pos + round(tEnd*dataRatios[j])), dtype=int))
        pos += tInd[j].size
    tOut = {'train':tInd[0], 'val':tInd[1], 'test':tInd[2]}
    return tOut

#==========================================================================
def PlotTrainMetrics(r, subplot=False):
    #Plot Training
    if not subplot:
        fig = plt.figure()
        fig.tight_layout()
#    ax = plt.gca()
    maxY = -9e9
    minY = 9e9
    
    lines = []
    lines.append({'label':'TrainSq',
                  'data':r.neutralTrainSqErr / r.trainMetrics.lossTrain,
                  'ls':'-', 'color':'C0'})
    lines.append({'label':'ValSq',
                  'data':r.neutralValSqErr / r.trainMetrics.lossVal,
                  'ls':'-', 'color':'C1'})
    lines.append({'label':'TrainAbs',
                  'data':r.neutralTrainAbsErr / r.trainMetrics.absErrTrain,
                  'ls':':', 'color':'C0'})
    lines.append({'label':'ValAbs',
                  'data':r.neutralValAbsErr / r.trainMetrics.absErrVal,
                  'ls':':', 'color':'C1'})
    handles = []
    for line in lines:
        lx, = plt.plot(line['data'], label=line['label'], linestyle=line['ls'], color=line['color'])
        handles.append(lx)
        maxY = max(maxY, max(line['data']))
        minY = min(minY, min(line['data']))
    plt.legend(handles = handles)
#    ax.set_yscale('log')
    
    if not subplot:
        plt.title('Training Scores (1=neutral, >1:better)')
        plt.show()
    
    return (maxY, minY)

#==========================================================================
def PrepConvConfig(r):
    """I made fairly flexible system for specifying the convolutional layer
    config. This function interprets the config and outputs explicit numbers for each layer.

    Args:
        r ([type]): [description]

    Returns:
        [dict]: convConf indicates the dilation, filter count and kernel size for each convolutional layer
    """
    convConf = dict()
    convConf['convDilation'] = r.config['convDilation']
    convConf['convFilters']  = r.config['convFilters']  
    convConf['convKernelSz'] = r.config['convKernelSz']

    # Check for zero layers
    if not (convConf['convDilation'] and convConf['convFilters'] and convConf['convKernelSz']):
        # at least 1 of these parameters are empty
        # there are no convolutional layers
        convLayerCount = 0
    else:
        convLayerCount = max(
            1 if isinstance(convConf['convDilation'], int) else len(convConf['convDilation']),
            1 if isinstance(convConf['convFilters'], int)  else len(convConf['convFilters']),
            1 if isinstance(convConf['convKernelSz'], int) else len(convConf['convKernelSz']),
        )
    
    convConf['layerCount'] = convLayerCount
    if isinstance(r.config['convDilation'], int):
        convConf['convDilation'] = [r.config['convDilation']] * convLayerCount
    if isinstance(r.config['convFilters'], int):
        convConf['convFilters'] = [r.config['convFilters']] * convLayerCount
    if isinstance(r.config['convKernelSz'], int):
        convConf['convKernelSz'] = [r.config['convKernelSz']] * convLayerCount
    
    return convConf


#==========================================================================
def MakeLayerModule(type:str, layer_input, out_width:int, dropout_rate:float=0., 
    kernel_size:int=None, dilation:int=None, stride:int=0, batch_norm:bool=True, name=None):
    # dropout
    # CNN or LSTM
    # avg pool with stride
    # batch normalization
    this_layer = layer_input

    if type.lower() == 'dense':
        if dropout_rate > 0.:
            this_layer = layers.Dropout(dropout_rate, name='do_' + name)(this_layer)
        this_layer = layers.Dense(units=out_width, activation='relu', name=name)(this_layer)
    elif type.lower() == 'conv':
        if dropout_rate > 0.:
            this_layer = layers.Dropout(dropout_rate, name='do_' + name)(this_layer)
        conv_args = {
            'filters' : out_width, # filter count = number of outputs
            'kernel_size' : kernel_size, # size of all filters
            'dilation_rate' : dilation, # factor in how far back to look
            # input_shape=(None, r.inFeatureCount),
            'use_bias' : True, 
            'padding' : 'causal', # causal; don't look into the future
            'activation' : 'relu',
            'name' : name
        }
        this_layer = layers.Conv1D(**conv_args)(this_layer)
    elif type.lower() == 'lstm':
        lstm_args = {
            'units' : out_width, # hidden layer size, & output size
            'dropout' : dropout_rate, # incorporated into the LSTM
            'activation' : 'tanh',
            'stateful' : False,
            'return_sequences' : True, # I'm including output values for all time steps, so always true
            'name' : name
        }
        this_layer = layers.LSTM(**lstm_args)(this_layer)
    else:
        raise Exception(f"MakeLayerModule type={type} is not recognized.")
    
    # avg pool with stride
    # to reduce data in temporal dimension
    if stride > 0.:
        this_layer = layers.AvgPool1D(pool_size=stride, strides=stride, padding='same', name='pl_' + name)(this_layer)
    
    # batch normalization

    this_layer = layers.BatchNormalization(name='bn_' + name)(this_layer)

    return this_layer


#==========================================================================
def MakeNetwork(r):
    # Prep convolution config
    convConf = PrepConvConfig(r)

    #Make a Neural Network
    if type(r.kerasOpt) == int:
        # beta_1 = exponential decay rate for 1st moment estimates. Default=0.9
        # beta_2 = exponential decay rate for 2nd moment estimates. Default=0.999
        opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9)
        
        r.optimiser = 'Adam'
    else:
        opt = r.kerasOpt
        r.optimiser = r.kerasOptStr

    feeds = [[] for i in range(FeedLoc.LEN)]
    
    # Keras functional API
    # Input feeds (applied at different locations)
    feeds[FeedLoc.conv] = layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.conv])), name='conv_feed')
    feeds[FeedLoc.lstm] = layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.lstm])), name='lstm_feed')
    feeds[FeedLoc.dense] = layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.dense])), name='dense_feed')
    feed_lens = [feeds[i].shape[-1] for i in range(FeedLoc.LEN)]

    # Make conv layers
    if feed_lens[FeedLoc.conv] > 0:
        convLayers = []
        for i in range(convConf['layerCount']):
            convLayers.append(MakeLayerModule('conv', feeds[FeedLoc.conv], out_width=convConf['convFilters'][i],
                kernel_size=convConf['convKernelSz'][i], dilation=convConf['convDilation'][i],
                dropout_rate=r.config['dropout'],
                name= f"conv1d_{i}_{convConf['convDilation'][i]}x"))

        if convConf['layerCount'] == 0:
            this_layer = feeds[FeedLoc.conv]
        elif convConf['layerCount'] == 1:
            this_layer = convLayers[0]
        elif convConf['layerCount'] > 1:
            this_layer = layers.concatenate(convLayers)

        # Add LSTM feed
        if feed_lens[FeedLoc.lstm] > 0:
            this_layer = layers.concatenate([this_layer, feeds[FeedLoc.lstm]], name='concat_bottleneck')
    else:
        # No convolutional input
        this_layer = feeds[FeedLoc.lstm]

    # Bottleneck layer (to reduce size going to LSTM)
    bnw = r.config['bottleneckWidth']
    if bnw > 0:
        this_layer = MakeLayerModule('dense', this_layer, out_width=bnw, dropout_rate=r.config['dropout'],
            name= f"bottleneck_{bnw}")

    # Make the LSTM layers
    for i, neurons in enumerate(r.config['lstmWidth']):
        if neurons > 0:
            this_layer = MakeLayerModule('lstm', this_layer, out_width=neurons, dropout_rate=r.config['dropout'],
                name= f'lstm_{neurons}')
    
    # Add dense feed
    if feed_lens[FeedLoc.dense] > 0:
        this_layer = layers.concatenate([this_layer, feeds[FeedLoc.dense]])

    # Dense layer
    main_output = layers.Dense(units=r.outFeatureCount, activation='linear', name='final_output')(this_layer)
    
    r.model = models.Model(inputs=feeds, outputs=[main_output])

    # mape = mean absolute percentage error
    r.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
    #r.model.build(input_shape=(None, r.inFeatureCount))
    
    r.trainMetrics = TrainMetrics()

    return

#==========================================================================
def PrintNetwork(r):
    r.model.summary()

    tf.keras.utils.plot_model(r.model, to_file='model.png', show_shapes=True)

    from IPython.display import Image, display
    img = Image('model.png')
    display(img)
    return

#==========================================================================
def TrainNetwork(r, inData, outData, final=True):
    """
    final == True indicates that this is the final call for TrainNetwork for
    this model.
    """

    r.tInd = _CalcIndices(r.timesteps, r.config['dataRatios'], r.config['excludeRecentDays'])
    
    #Callbacks
    callbacks = []
    
    # Callback to validate data
    validationCb = ValidationCb()
    valI = r.tInd['val'] # Validation indices
    startPredict = max(0, valI.min()-100) # This number of time steps are used to build state before starting predictions

    valInData = [arr[:,startPredict:valI.max()+1,:] for arr in inData]
    validationCb.setup(valInData, outData[:,valI,:], r.trainMetrics, r.config['earlyStopping'], r.config['epochs'])
    
    # The model could be partially trained
    epochsLeft = r.config['epochs'] - r.trainMetrics.curEpoch
    if epochsLeft == 0:
        print('\r\n\n\nERROR! NO EPOCHS REMAINING ON TRAINING!')
        return
    
    
    callbacks.append(validationCb)
    
    # Save best model
#    fileBestWeights = "bestModel.h5"
#    checkpoint = keras.callbacks.ModelCheckpoint(fileBestWeights,
#                                                 monitor='fitness', verbose=0, save_best_only=True,  mode='max')
#    callbacks.append(checkpoint)

#    callbacks += [keras.callbacks.TensorBoard(log_dir='./logs2', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)]
    if r.trainMetrics.curEpoch == 0:
        print(f"\nStarting training. Max {r.config['epochs']} epochs")
    else:
        print(f"\nStarting training. At epoch {r.trainMetrics.curEpoch}. Max {r.config['epochs']} epochs. {epochsLeft} remaining.")
        
    
    trainX = [arr[:,r.tInd['train'],:] for arr in inData]
    trainY = outData[:, r.tInd['train']]
    valY = outData[:, r.tInd['val']]
    
    r.neutralTrainAbsErr = np.sum(np.abs(trainY)) / trainY.size
    r.neutralValAbsErr = np.sum(np.abs(valY)) / valY.size
    r.neutralTrainSqErr = np.sum(np.abs(trainY)**2) / trainY.size
    r.neutralValSqErr = np.sum(np.abs(valY)**2) / valY.size
    
    start = time.time()

    validationCb.startTime = time.time()
    hist = r.model.fit(trainX, trainY, epochs=epochsLeft, 
                     batch_size=r.sampleCount, shuffle=True,
                     verbose=0, callbacks=callbacks)
    
    end = time.time()
    r.trainTime = end-start
    print(f'Training Time (h:m:s)= {SecToHMS(r.trainTime)}.  {r.trainTime:.1f}s')
    
    if final and r.config['revertToBest']:
        if validationCb.bestEpoch > 0:
            print(f'Reverting to the model with best validation (epoch {validationCb.bestEpoch})')
            r.model.set_weights(validationCb.bestWeights)
    
    PlotTrainMetrics(r)

    return


#==========================================================================
def MakeAndTrainNetwork(r, inData, outData):
    MakeNetwork(r)
    PrintNetwork(r)
    TrainNetwork(r, inData, outData)
    return

#==========================================================================
# Make several networks and choose the best
def MakeAndTrainPrunedNetwork(r, inData, outData):
    # SETTINGS
    candidates = 5
    trialEpochs = 16

    # Create all models
    models = [0] * candidates
    trainMetrics = [0] * candidates
    for i in range(candidates):
        MakeNetwork(r)
        models[i] = r.model
        trainMetrics[i] = r.trainMetrics
        if i == 0:
            PrintNetwork(r)
    printmd('**********************************************************************************')
    printmd('## PRUNED NETWORK')
    printmd(f'Training **{candidates}** candidate models for **{trialEpochs}** epochs, then selecting the best.')

    # Trial each model by a small amount of training
    epochBackup = r.config['epochs']
    r.config['epochs'] = trialEpochs
    lossTrain = np.zeros((candidates))
    lossVal = np.zeros((candidates))
    trainGradSum = np.zeros((candidates))
    valGradSum = np.zeros((candidates))
    fitness = np.zeros((candidates))
    for i in range(candidates):
        print('\n************************************')
        printmd(f'Training candidate model **{i}** out of {candidates}')
        r.model = models[i]
        r.trainMetrics = trainMetrics[i]
        TrainNetwork(r, inData, outData, final=False)
        
        lossTrain[i] = r.trainMetrics.lossTrain[-1]
        lossVal[i] = r.trainMetrics.lossVal[-1]
        fitness[i] = np.max(r.trainMetrics.fitness)
        # gradSum is a sum of the last 5 gradients (from the last 6 values)
        # of the log of the loss.
        # Should be negative. More negative = better
        # The most recent is weighted more than the first
        pastVal = min(6,trialEpochs)
        trainGradSum[i] = np.sum(np.diff(np.log(r.trainMetrics.lossTrain[-pastVal:])) * np.linspace(1,2,num=pastVal-1))
        valGradSum[i] = np.sum(np.diff(np.log(r.trainMetrics.lossVal[-pastVal:])) * np.linspace(1,2,num=pastVal-1))
       
  
    # PICK THE BEST MODEL
    # Higher score = better
    scores = pd.DataFrame()
    scores['train'] = lossTrain.min()/lossTrain # 0 to 1 (1 being the best candidate)
    scores['val'] = fitness/fitness.max() # 0 to 1 (1 being the best candidate)
    # For gradient scores, 1 is the top score, and it scales down from there
    # The amount that it drops is determined by 

    # Method prior to 2021-11-07:
    # trainGradScale = np.abs(np.mean(np.log(lossTrain)))
    # temp = trainGradSum / trainGradScale * 8
    # scores['trainGrad'] = np.clip(temp.min()-temp+1, -1, 1)

    scores['trainGrad'] = np.clip(-(trainGradSum / np.abs(trainGradSum.min())), -1, 1)
    scores['valGrad'] = np.clip(-(valGradSum / np.abs(valGradSum.min())), -1, 1)

    print('\n**********************************************************************************')
    print('**********************************************************************************')
    print('**********************************************************************************')
    print('### All candidate model scores:')
    print(scores)
    
    # Weight each of the scores
    scores['train'] *= 1
    scores['val'] *= 2
    scores['trainGrad'] *= 0.5
    scores['valGrad'] *= 0.5
    
    totalScore = scores.sum(axis=1)
              
    bestI = totalScore.argmax()
     
    print('Total:')
    print(totalScore)
    printmd('**Chose candidate model: {}**'.format(bestI))
    
    # Train on the best model
    r.config['epochs'] = epochBackup
    r.model = models[bestI]
    r.trainMetrics = trainMetrics[bestI]
    TrainNetwork(r, inData, outData)
    return
    
#==========================================================================
def TestNetwork(r, priceData, inData, outData):
    tPlot = np.r_[0:r.timesteps] # Range of output plot (all data)
    if (r.config['dataRatios'][2] > 0.1):
        print('WARNING! TestNetwork uses Val Data as the test data, but Test Data also exists. ')
        print(r.config['dataRatios'])
    testI = r.tInd['val'] # Validation indices used as test
    
    #Predictions (entire input data range)
    predictY = r.model.predict(inData, batch_size=r.sampleCount)
    
    def _PlotOutput(priceData, out, predict, tRange, sample):
        """Plot a single output feature of 1 sample"""
        plotsHigh = 1+r.outFeatureCount
        fig, axs = plt.subplots(plotsHigh, 1, sharex=True, figsize=(12,3*plotsHigh))
        fig.tight_layout()
        
        ax = axs[0]
        ax.figure = fig # required to avoid an exception
        ax.semilogy(tRange, priceData[sample, tRange]) # Daily data
        ax.set_title('Prices. Sample {} ({}) [{}]'.format(r.coinList[sample], sample, r.batchRunName))
        
        for feature in range(r.outFeatureCount):
            ax = axs[1+feature]
            ax.figure = fig
            predictYPlot = predict[sample, :, feature]
            outPlot = out[sample, tRange, feature]
            l1, = ax.plot(tRange, outPlot, label='Actual')
            l2, = ax.plot(tRange, predictYPlot, label='Prediction')
            l3, = ax.plot([r.tInd['train'][0], r.tInd['train'][0]], [np.min(outPlot), np.max(outPlot)], label='TrainStart')
            l4, = ax.plot([r.tInd['train'][-1], r.tInd['train'][-1]], [np.min(outPlot), np.max(outPlot)], label='TrainEnd')
            l0, = ax.plot([tRange[0], tRange[-1]], [0, 0])
            ax.set_title('Output Feature {} ({}-{}steps)'.format(feature, r.config['outputRanges'][feature][0], r.config['outputRanges'][feature][1]))
            ax.legend(handles = [l1, l2, l3 , l4])
        # Save file if necessary
        if r.isBatch and sample == 0:
            try:
                directory = './' + r.batchName
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + '/'+ r.batchRunName +'.png'
                plt.savefig(filename.replace(':','-'))
            except:
                print('\n\n SAVING FILE ERROR')
        plt.show()
    
    r.prediction = predictY
    # Plot prediction
    for s in range(r.sampleCount):
        _PlotOutput(priceData, outData, predictY, tPlot, s)
    
    r.testAbsErr = np.sum(np.abs(predictY[:,testI,:] - outData[:,testI,:])) / predictY[:,testI,:].size
    r.neutralTestAbsErr = np.sum(np.abs(outData[:,testI,:])) / outData[:,testI,:].size
    r.testScore = r.neutralTestAbsErr / r.testAbsErr
    
    r.trainAbsErr = np.sum(np.abs(predictY[:,r.tInd['train'],:] - outData[:,r.tInd['train'],:])) / predictY[:,r.tInd['train'],:].size
    r.trainScore = r.neutralTrainAbsErr / r.trainAbsErr
    
    # Assess the level of movement (some networks don't train and the result
    # is just a straight line)
    
    
    # Assess whether or not a 'floor' is occurring - if a large percent of the
    # data is close to the minimum
    
    print('Scores (1:neutral, >1 :better than neutral)')
    print('Train Score: {:.3f}\nTest Score: {:.3f} '.format(r.trainScore, r.testScore))
    return
