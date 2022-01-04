# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:12:18 2017

@author: Dean
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers

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
class FitnessCb(tf.keras.callbacks.Callback):
    """
    This callback is used to validate/test the model after each epoch. 
    """
    def __init__(self):
        pass
    
    
    def setup(self, inData, outTarget, neutralTrainSqErr, neutralValSqErr, neutralTrainAbsErr, neutralValAbsErr):
        # inData covers all of the time steps of outData, PLUS some earlier time steps
        # these earlier timesteps are used for building state
        self.inData = inData # The input data that will be used for validation
        self.outTarget = np.array(outTarget) # The target output. A perfect prediction model would predict these values
        self.targetSize = outTarget.size # Number of points
        self.targetTimeSteps = outTarget.shape[-2]

        self.neutralTrainAbsErr = neutralTrainAbsErr # train error if output was 'always neutral'
        self.neutralTrainSqErr = neutralTrainSqErr # train error if output was 'always neutral'
        self.neutralValAbsErr = neutralValAbsErr
        self.neutralValSqErr = neutralValSqErr
        
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

           
    #==========================================================================
    def on_epoch_end(self, epoch, logs={}):
        fitness = 1/logs['val_mean_squared_error']
        penalty = 1. # fitness is multiplied by this penalty
        

        if 0: # !@# I've currently disabled the penalty calculations as it's expensive to predict
            # Ideally, I'd be able to access predictions from the callback, but keras doesn't
            # allow that. I could write a metric to perform this calculation, but that's a
            # challenge to use onlly 'tensor operations'
            predictY = self.model.predict(self.inData, batch_size=100)
            # Note that the prediction has some initial period to build up state,
            # then the actual prediction (len=targetTimeSteps)
            err = predictY[:,-self.targetTimeSteps:,:] - self.outTarget
            
            # Penalise predictions that don't vary across the time series
            thisDiff = np.mean(np.abs(np.diff(predictY, axis=1)))
            debug = 0
            if debug: print('Dif Score {:5f}, Lower {:5f}, Upper {:5f}'.format(thisDiff, self.avgDiffLower, self.avgDiffUpper), end='')
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
        logs['penalty'] = penalty

        logs['train_score_abs'] = self.neutralTrainAbsErr / logs['mean_absolute_error']
        logs['val_score_abs'] = self.neutralValAbsErr / logs['val_mean_absolute_error']
        logs['train_score_sq'] = self.neutralTrainSqErr / logs['mean_squared_error']
        logs['val_score_sq'] = self.neutralValSqErr / logs['val_mean_squared_error']


#==========================================================================
class CheckpointCb(tf.keras.callbacks.Callback):
    """
    Performs early stopping
    I could alternatively use the keras built-in method: keras.callbacks.EarlyStopping
    venv\Lib\site-packages\tensorflow\python\keras\callbacks.py
    """
    def __init__(self):
        pass
    
    
    def setup(self, patience):
        # Tracking the best result:
        self.bestFitness = 0
        self.bestWeights = []
        self.bestEpoch = 0
        # Early stopping
        self.wait = 0
        self.patience = patience # Epochs before stopping early. Set to 0 to disable early stopping

        self.prevFitness = 0
        self.stopped_epoch = 0

           
    #==========================================================================
    def on_epoch_end(self, epoch, logs={}):

        fitness = logs['fitness']

        # Check if this is the best result
        bestResult = False
        if fitness > self.bestFitness:
            self.bestFitness = fitness
            self.bestWeights = self.model.get_weights()
            self.bestEpoch = epoch
            bestResult = True

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
                    print('STOPPING TRAINING EARLY AT EPOCH {}'.format(epoch))
        
        logs['newBest'] = bestResult
        self.prevFitness = fitness
    

#==========================================================================
class PrintoutCb(tf.keras.callbacks.Callback):
    """
    This callback prints info
    """
    def __init__(self):
        pass
    
    
    def setup(self, maxEpochs):
        self.maxEpochs = maxEpochs

        self.startTime = time.time()
        self.prevPrintTime = time.time()
        self.printCount = 0 # the number of times we've printed
        self.epochTimeHist = [] # a list of the previous ~3 epoch training durations
           
    #==========================================================================
    def on_epoch_end(self, epoch, logs={}):
        if not self.epochTimeHist:
            self.epochTimeHist.append(self.startTime)

        # add the epoch duration to the list
        now = time.time()
        self.epochTimeHist.append(now)

        epochsRemaining = self.maxEpochs - epoch
        timePerEpoch = np.mean(np.diff(self.epochTimeHist[-4:]))
        timeRemaining = epochsRemaining * timePerEpoch

        def PrintHeaders():
            # Headers for the text table printed during training
            print(f"Epoch TrainAbsSc ValAbsSc Fitness ProcTime Remaining")


        # Epoch printout
        if logs['newBest'] or (epoch%10)==0 or now - self.prevPrintTime > 60. \
            or epoch < 5 or epoch+1 == self.maxEpochs:
            if self.printCount%10 == 0:
                PrintHeaders()
            #print(f"Epoch {epoch:2} - TrainErrSq={logs['loss']:6.3f}, ValErrSq={val_sq_err:6.3f}, Fitness={fitness:6.3f}, " +
                 #f"Penalty= {penalty:5.3f} {' - New best! ' if bestResult else ''}")
            print(f"{epoch:5} " +
            f"{logs['train_score_abs']:10.3f} " +
            f"{logs['val_score_abs']:8.3f} " +
            f"{logs['fitness']:7.3f} " +
            #f"{logs['penalty']:7.3f} " +
            f"{(now - self.epochTimeHist[-2]):7.1f}s {SecToHMS(timeRemaining):>9s}" + 
            f"{' - New best! ' if logs['newBest'] else ''}")
            self.printCount += 1
            self.prevPrintTime = now

    
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
def PlotTrainMetrics(r, axIn=None):
    #Plot Training
    if axIn is None:
        fig, ax = plt.subplots()
        fig.tight_layout()
    else:
        ax = axIn

    maxY = -9e9
    minY = 9e9
    
    lines = []
    lines.append({'label':'TrainSq',
                  'data':r.trainHistory['train_score_sq'],
                  'ls':'-', 'color':'C0'})
    lines.append({'label':'ValSq',
                  'data':r.trainHistory['val_score_sq'],
                  'ls':'-', 'color':'C1'})
    lines.append({'label':'TrainAbs',
                  'data':r.trainHistory['train_score_abs'],
                  'ls':':', 'color':'C0'})
    lines.append({'label':'ValAbs',
                  'data':r.trainHistory['val_score_abs'],
                  'ls':':', 'color':'C1'})
    handles = []
    for line in lines:
        lx, = ax.plot(line['data'], label=line['label'], linestyle=line['ls'], color=line['color'])
        handles.append(lx)
        maxY = max(maxY, max(line['data']))
        minY = min(minY, min(line['data']))
    ax.legend(handles = handles)
#    ax.set_yscale('log')
    
    if axIn is None:
        ax.set_title('Training Scores (1=neutral, >1:better)')
        fig.show()
    
    return (maxY, minY)

#==========================================================================
def PrepConvConfig(cfg):
    """I made fairly flexible system for specifying the convolutional layer
    config. This function interprets the config and outputs explicit numbers for each layer.

    Args:
        r ([type]): [description]

    Returns:
        [dict]: convCfg indicates the dilation, filter count and kernel size for each convolutional layer
    """
    convCfg = dict()
    convCfg['dilation'] = cfg['convDilation']
    convCfg['filters']  = cfg['convFilters']  
    convCfg['kernelSz'] = cfg['convKernelSz']

    # Determine the number of convolutional layers
    if not (convCfg['dilation'] and convCfg['filters'] and convCfg['kernelSz']):
        # Check for zero layers
        # at least 1 of these parameters are empty
        # there are no convolutional layers
        convLayerCount = 0
    else:
        convLayerCount = max(
            1 if isinstance(convCfg['dilation'], int) else len(convCfg['dilation']),
            1 if isinstance(convCfg['filters'], int)  else len(convCfg['filters']),
            1 if isinstance(convCfg['kernelSz'], int) else len(convCfg['kernelSz']),
        )
    
    for key in convCfg.keys():
        if isinstance(convCfg[key], int) or convLayerCount==0:
            convCfg[key] = [convCfg[key]] * convLayerCount

    convCfg['layerCount'] = convLayerCount
    
    return convCfg


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
    convCfg = PrepConvConfig(r.config)

    #Make a Neural Network
    if r.config['optimiser'].lower() == 'adam':
        # beta_1 = exponential decay rate for 1st moment estimates. Default=0.9
        # beta_2 = exponential decay rate for 2nd moment estimates. Default=0.999
        opt = optimizers.Adam(learning_rate=r.config['learningRate'], beta_1=0.9)
    else:
        opt = r.config['optimiser'].lower()

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
        for i in range(convCfg['layerCount']):
            convLayers.append(MakeLayerModule('conv', feeds[FeedLoc.conv], out_width=convCfg['filters'][i],
                kernel_size=convCfg['kernelSz'][i], dilation=convCfg['dilation'][i],
                dropout_rate=r.config['dropout'],
                name= f"conv1d_{i}_{convCfg['dilation'][i]}x"))

        if convCfg['layerCount'] == 0:
            this_layer = feeds[FeedLoc.conv]
        elif convCfg['layerCount'] == 1:
            this_layer = convLayers[0]
        elif convCfg['layerCount'] > 1:
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

    # LSTM layers
    for i, neurons in enumerate(r.config['lstmWidths']):
        if neurons > 0:
            this_layer = MakeLayerModule('lstm', this_layer, out_width=neurons, dropout_rate=r.config['dropout'],
                name= f'lstm_{i}_{neurons}')
    
    # Add dense feed
    if feed_lens[FeedLoc.dense] > 0:
        this_layer = layers.concatenate([this_layer, feeds[FeedLoc.dense]])

    # Dense layer
    main_output = layers.Dense(units=r.outFeatureCount, activation='linear', name='final_output')(this_layer)
    
    r.model = CustomModel(inputs=feeds, outputs=[main_output])
    r.modelEpoch = -1
    r.trainHistory = {}

    # mape = mean absolute percentage error
    r.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error', 'mean_squared_error'])
    #r.model.build(input_shape=(None, r.inFeatureCount))

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
def PrepTrainNetwork(r, inData, outData) -> dict :
        # Generate validation data
    r.tInd = _CalcIndices(r.timesteps, r.config['dataRatios'], r.config['excludeRecentSteps'])
    
    valI = r.tInd['val'] # Validation indices
    startPredict = max(0, valI[0]-r.config['evaluateBuildStatePoints']) # This number of time steps are used to build state before starting predictions
    valX = [arr[:, startPredict:valI[-1]+1, :] for arr in inData]
    valY = outData[:,valI,:]

    if r.modelEpoch == -1:
        print(f"\nStarting training. Max {r.config['epochs']} epochs")
    else:
        print(f"\nStarting training. At epoch {r.modelEpoch+1}. Max {r.config['epochs']} epochs. {r.config['epochs'] - r.modelEpoch -1} remaining.")

    trainX = [arr[:,r.tInd['train'],:] for arr in inData]
    trainY = outData[:, r.tInd['train']]
    valY = outData[:, r.tInd['val']]
    
    r.neutralTrainAbsErr = np.sum(np.abs(trainY)) / trainY.size
    r.neutralValAbsErr = np.sum(np.abs(valY)) / valY.size
    r.neutralTrainSqErr = np.sum(np.abs(trainY)**2) / trainY.size
    r.neutralValSqErr = np.sum(np.abs(valY)**2) / valY.size

    #Callbacks
    callbacks = []

    fitnessCb = FitnessCb()
    fitnessCb.setup(valX, valY, r.neutralTrainSqErr, r.neutralValSqErr, r.neutralTrainAbsErr, r.neutralValAbsErr)
    callbacks.append(fitnessCb)

    checkpointCb = CheckpointCb()
    checkpointCb.setup(r.config['earlyStopping'])
    callbacks.append(checkpointCb)

    printoutCb = PrintoutCb()
    printoutCb.setup(r.config['epochs'])
    callbacks.append(printoutCb)

    fitArgs = {
        'x':trainX,
        'y':trainY,
        'epochs':r.config['epochs'],
        'validation_data':(valX, valY),
        'batch_size':r.sampleCount,
        'shuffle':True,
        'verbose':0,
        'callbacks':callbacks,
        'initial_epoch':r.modelEpoch+1
    }

    return fitArgs, checkpointCb, printoutCb
    


#==========================================================================
def TrainNetwork(r, inData, outData, final=True):
    """
    final == True indicates that this is the final call for TrainNetwork for
    this model.
    """

    # Pre-fit tasks
    fitArgs, checkpointCb, printoutCb = PrepTrainNetwork(r, inData, outData)
    
    # FIT
    start = time.time()
    printoutCb.startTime = start
    hist = r.model.fit(**fitArgs)

    # Post-fit tasks
    if r.modelEpoch == -1:
        r.trainHistory = hist.history
    else:
        # When 'reverting', the model epoch jumps backwards
        # Overwrite the 'reverted' section of train metrics, and append the new
        for key in hist.history.keys():
            r.trainHistory[key][hist.epoch[0]:hist.epoch[-1]+1] = hist.history[key]
 

    if hist.epoch[-1]+1 != len(list(r.trainHistory.values())[0]):
        raise Exception("Cur epoch doesn't match training hist. Program error")
    
    end = time.time()
    r.trainTime = end-start
    print(f'Training Time (h:m:s)= {SecToHMS(r.trainTime)}.  {r.trainTime:.1f}s')
    
    r.modelEpoch = hist.epoch[-1]

    # Model reverting
    if final and r.config['revertToBest']:
        if checkpointCb.bestEpoch not in [0, r.modelEpoch]:
            print(f'Reverting to the model with best validation (epoch {checkpointCb.bestEpoch})')
            r.model.set_weights(checkpointCb.bestWeights)
            # Note that r.trainHistory history for the full training
            r.modelEpoch = checkpointCb.bestEpoch
    
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
    trainHist = [0] * candidates
    for i in range(candidates):
        MakeNetwork(r)
        models[i] = r.model
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
        TrainNetwork(r, inData, outData, final=False)
        trainHist[i] = r.trainHistory

        lossTrain[i] = r.trainHistory['mean_squared_error'][-1]
        lossVal[i] = r.trainHistory['val_mean_squared_error'][-1]
        # gradSum is a sum of the last 5 gradients (from the last 6 values)
        # of the log of the loss.
        # Should be negative. More negative = better
        # The most recent is weighted more than the first
        pastVal = min(6,trialEpochs)
        trainGradSum[i] = np.sum(np.diff(np.log(r.trainHistory['mean_squared_error'][-pastVal:])) * np.linspace(1,2,num=pastVal-1))
        valGradSum[i] = np.sum(np.diff(np.log(r.trainHistory['val_mean_squared_error'][-pastVal:])) * np.linspace(1,2,num=pastVal-1))
       
  
    # PICK THE BEST MODEL
    # Higher score = better
    scores = pd.DataFrame()
    scores['train'] = lossTrain.min()/lossTrain # 0 to 1 (1 being the best candidate)
    scores['val'] = lossVal/lossVal.max() # 0 to 1 (1 being the best candidate)
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
    r.trainHistory = trainHist[bestI]
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
        fig, axs = plt.subplots(plotsHigh, 1, sharex=True, figsize=(r.config['plotWidth'],3*plotsHigh))
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

#==========================================================================
from tensorflow.python.keras.engine import data_adapter
class CustomModel(tf.keras.Model):
  def test_step(self, data):
    """The logic for one evaluation step.
    Overridden by Dean to allow for the val_x to have more timesteps than
    val_y. In this case, y_pred is truncated to the length of val_y, cutting
    off the initial entries.
    The purpose here is that the start of the prediction is used for
    building state and not for evaluation. That section can overlap with 
    the training set as it's not used for evaluation.

    This method can be overridden to support custom evaluation logic.
    This method is called by `Model.make_test_function`.

    This function should contain the mathematical logic for one step of
    evaluation.
    This typically includes the forward pass, loss calculation, and metrics
    updates.

    Configuration details for *how* this logic is run (e.g. `tf.function` and
    `tf.distribute.Strategy` settings), should be left to
    `Model.make_test_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned.
    """
    # Unpack the data
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    # Compute predictions
    y_pred = self(x, training=False)
    # CUSTOM LINE: shrink y_pred to the size of y
    # (x can have more timesteps than y)
    y_pred = y_pred[:, -y.shape[-2]:, :] # samples, timesteps, features
    # END CUSTOM LINE
    # Updates stateful loss metrics.
    self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)
    # Update the metrics
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    # Collect metrics to return
    return_metrics = {}
    for metric in self.metrics:
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics
