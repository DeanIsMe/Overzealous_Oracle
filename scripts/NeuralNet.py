# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:12:18 2017

@author: Dean
"""

from re import A
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.regularizers as regularizers

import pandas as pd
import time
import os
#from ClockworkRNN import CWRNN

from DataTypes import printmd, SecToHMS, FeedLoc
from Config_CC import GetConfig, PrintConfigString

#==========================================================================
class RegularizerDef:
    """Defines parameters for a keras regularizer
    This class exists to generate kernel regularizers concisely with make_reg
    I COULD just re-use the same regularizer object, as suggested here:
    # https://stackoverflow.com/a/58871203/3580080
    But generating a new object each time is seen as better practice
    """
    def __init__(self, type, rate_1, rate_2):
        """Give all parameters here
        Args:
            type (_type_): l1, l2, or l1_l2
            factor_a (_type_): regularization rate for l1
            factor_b (_type_): regularization rate for l2
        """
        self.type = type
        self.rate_1 = rate_1
        self.rate_2 = rate_2


def make_reg(reg_def):
    if reg_def is None:
        return None
    if reg_def.type is None:
        return None
    elif reg_def.type.lower() == 'l1':
        return regularizers.l1(reg_def.rate_1)
    elif reg_def.type.lower() == 'l2':
        return regularizers.l2(reg_def.rate_2)
    elif reg_def.type.lower() == 'l1_l2':
        return regularizers.l1_l2(l1=reg_def.rate_1, l2=reg_def.rate_2)
    else:
        raise Exception(f"Invalid regularizer type: {reg_def.type}")


#==========================================================================
class FitnessCb(tf.keras.callbacks.Callback):
    """
    This callback is used to validate/test the model after each epoch. 
    """
    def __init__(self):
        pass
    
    
    def setup(self, trainX, trainY, valX, valY, neutralTrainSqErr, neutralValSqErr, neutralTrainAbsErr, neutralValAbsErr):
        # valX covers all of the time steps of outData, PLUS some earlier time steps
        # these earlier timesteps are used for building state

        self.neutralTrainAbsErr = neutralTrainAbsErr # train error if output was 'always neutral'
        self.neutralTrainSqErr = neutralTrainSqErr # train squared error if output was 'always neutral'
        self.neutralValAbsErr = neutralValAbsErr
        self.neutralValSqErr = neutralValSqErr
        
        # Determine the thresholds for penalising lack of movement
        # This is to avoid uneventful results that don't do anything
        self.trainDynamismTarget = np.mean(np.abs(trainY[:, :-10, :] - trainY[:, 10:, :]))
        self.valDynamismTarget = np.mean(np.abs(valY[:, :-10, :] - valY[:, 10:, :]))

           
    #==========================================================================
    def on_epoch_end(self, epoch, logs={}):
        # Simple scores
        logs['train_score_abs'] = self.neutralTrainAbsErr / logs['mean_absolute_error']
        logs['val_score_abs'] = self.neutralValAbsErr / logs['val_mean_absolute_error']
        logs['train_score_sq'] = self.neutralTrainSqErr / logs['mean_squared_error']
        logs['val_score_sq'] = self.neutralValSqErr / logs['val_mean_squared_error']

        # PENALTY & FITNESS
        # CONSTANTS
        UPPER_R = 0.25 # above this, there's no penalisation
        LOWER_R = 0.1 # Below this, the penalty is a maximum
        PENALTY_LOWER = 0.01 # Max penalty (scaler)
        PENALTY_UPPER = 1. # Always 1
        
        # Penalise predictions that don't vary across the time series

        def CalcPenalty(dynamismRatio):
            # dynamismRatio is how 'dynamic' the prediction is compared to the target output
            # return penalty. fitness is multiplied by this penalty
            if dynamismRatio >= UPPER_R:
                return 1.0

            pos = (dynamismRatio - LOWER_R) / (UPPER_R - LOWER_R) # 0 to 1
            penalty = (pos) * (PENALTY_UPPER - PENALTY_LOWER) + PENALTY_LOWER
            penalty = np.clip(penalty, PENALTY_LOWER, PENALTY_UPPER)
            return penalty
            
        logs['penalty'] = CalcPenalty(trainDynamismRatio:= logs['dynamism'] / self.trainDynamismTarget)
        logs['val_penalty'] = CalcPenalty(valDynamismRatio:= logs['val_dynamism'] / self.valDynamismTarget)

        logs['fitness'] = logs['score_sq_any'] * logs['penalty']
        logs['val_fitness'] = logs['val_score_sq_any'] * logs['val_penalty']
       
        debug = 0
        if debug: print(f"valDynamismRatio   = {valDynamismRatio:7.5f},  valPenalty = {logs['val_penalty']:5f}")
        if debug: print(f"trainDynamismRatio = {trainDynamismRatio:7.5f},  trainPenalty = {logs['penalty']:5f}")




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

        fitness = logs['val_fitness']

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
            print(f"Epoch TrainSqSc TrainScAny ValSqSc ValScAny ValPenalty ValFitness ProcTime Remaining")


        # Epoch printout
        if logs['newBest'] or (epoch%10)==0 or now - self.prevPrintTime > 60. \
            or epoch < 5 or epoch+1 == self.maxEpochs:
            if self.printCount%10 == 0:
                PrintHeaders()
            #print(f"Epoch {epoch:2} - TrainErrSq={logs['loss']:6.3f}, ValErrSq={val_sq_err:6.3f}, Fitness={fitness:6.3f}, " +
                 #f"Penalty= {penalty:5.3f} {' - New best! ' if bestResult else ''}")
            print(f"{epoch:5} " +
            f"{logs['train_score_sq']:9.3f} " +
            f"{logs['score_sq_any']:10.3f} " +
            f"{logs['val_score_sq']:7.3f} " +
            f"{logs['val_score_sq_any']:8.3f} " +
            f"{logs['val_penalty']:10.3f} " +
            f"{logs['val_fitness']:10.3f} " +
            f"{(now - self.epochTimeHist[-2]):7.2f}s {SecToHMS(timeRemaining):>9s}" + 
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
def PlotTrainMetrics(trainHistory, axIn=None, legend=True, plotWidth=7):
    #Plot Training
    if axIn is None:
        fig, ax = plt.subplots(figsize=(plotWidth, 4))
        fig.tight_layout()
    else:
        ax = axIn

    dataPoints = len(list(trainHistory.values())[0])
    maxY = -9e9
    minY = 9e9
    
    lines = []
    lines.append({'label':'Neutral',
                  'data':np.array([1.0] * dataPoints),
                  'ls':'-', 'color':'k', 'lw':1})

    lines.append({'label':'TrainScoreAny',
                  'data':trainHistory['score_sq_any'],
                  'ls':':', 'color':'C0', 'lw':2.5})
    lines.append({'label':'TrainPenalty',
                  'data':trainHistory['penalty'],
                  'ls':'--', 'color':'C0', 'lw':1.5})
    lines.append({'label':'TrainFitness',
                  'data':trainHistory['fitness'],
                  'ls':'-', 'color':'C0', 'lw':1.5})


    lines.append({'label':'ValScoreAny',
                  'data':trainHistory['val_score_sq_any'],
                  'ls':':', 'color':'C1', 'lw':2.5})
    lines.append({'label':'ValPenalty',
                  'data':trainHistory['val_penalty'],
                  'ls':'--', 'color':'C1', 'lw':1.5})
    lines.append({'label':'ValFitness',
                  'data':trainHistory['val_fitness'],
                  'ls':'-', 'color':'C1', 'lw':1.5})

    handles = []
    for line in lines:
        lx, = ax.plot(line['data'], label=line['label'], linestyle=line['ls'], color=line['color'], lw=line['lw'])
        handles.append(lx)
        maxY = max(maxY, max(line['data']))
        minY = min(minY, min(line['data']))
    if legend:
        ax.legend(handles = handles)
#    ax.set_yscale('log')
    ax.set_xlim(left=0, right=dataPoints-1)
    ax.grid(True)
    
    if axIn is None:
        ax.set_title('Training Scores (1=neutral, >1:better)')
        plt.show()
    
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
    kernel_size:int=None, dilation:int=None, stride:int=0, 
    batch_norm:bool=True, k_reg=None, name=None):
    # dropout
    # CNN or LSTM
    # avg pool with stride
    # batch normalization
    this_layer = layer_input

    if type.lower() == 'dense':
        if dropout_rate > 0.:
            this_layer = layers.Dropout(dropout_rate, name='do_' + name)(this_layer)
        this_layer = layers.Dense(units=out_width, kernel_regularizer=k_reg, activation='relu', name=name)(this_layer)
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
            'kernel_regularizer' : k_reg,
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
            'kernel_regularizer' : k_reg,
            'name' : name
        }
        this_layer = layers.LSTM(**lstm_args)(this_layer)
    elif type.lower() == 'gru':
        gru_args = {
            'units' : out_width, # hidden layer size, & output size
            'dropout' : dropout_rate, # incorporated into the LSTM
            'activation' : 'tanh',
            'stateful' : False,
            'return_sequences' : True, # I'm including output values for all time steps, so always true
            'kernel_regularizer' : k_reg,
            'name' : name
        }
        this_layer = layers.GRU(**gru_args)(this_layer)
    else:
        raise Exception(f"MakeLayerModule type={type} is not recognized.")
    
    # avg pool with stride
    # to reduce data in temporal dimension
    if stride > 0.:
        this_layer = layers.AvgPool1D(pool_size=stride, strides=stride, padding='same', name='pl_' + name)(this_layer)
    
    # batch normalization
    if batch_norm:
        this_layer = layers.BatchNormalization(name='bn_' + name)(this_layer)

    return this_layer


#==========================================================================
def mean_squared_error_any(y_true, y_pred):
    """Custom keras metric function
    Chooses the output feature that performed the best, and returns the 
    mean squared error for that (ignoring other output features)
    """
    se = tf.square(tf.subtract(y_pred, y_true))
    mse = tf.reduce_mean(se, axis=-2) # Average across timesteps
    return tf.reduce_min(mse, axis=-1) 

#==========================================================================
def score_sq_any(y_true, y_pred):
    """Custom keras metric function
    Chooses the output feature that performed the best, and returns the 
    score for that (ignoring other output features)
    Note that this is somewhat inefficient because it recalculates the
    neutral squared error each time.
    I COULD calculate the neutral score once and save it, then assume it will be the same for future calls.
    """
    seNeutral = tf.square(y_true)
    seNeutral = tf.reduce_mean(seNeutral, axis=[-2, 0]) # Average across timesteps & samples

    se = tf.square(tf.subtract(y_pred, y_true))
    se = tf.reduce_mean(se, axis=[-2, 0]) # Average across timesteps & samples

    # Calculate the score for each
    scores = tf.divide(seNeutral, se)
    return tf.reduce_max(scores, axis=-1) # Choose the best score from the different output features

#==========================================================================
def dynamism(y_true, y_pred):
    """Custom keras metric function
    Assesses how much the prediction changes
    y_true and y_pred shape is (batch_size, timesteps, features)
    Return shape is not important, because Keras reduces the output to a single value (by averaging).
    """
    return tf.abs(tf.subtract(y_pred[:,:-10,:], y_pred[:,10:,:]))


#==========================================================================
def MakeNetwork(r):
    # Prep convolution config
    convCfg = PrepConvConfig(r.config)
    
    reg_def = RegularizerDef(r.config['regularizerType'], r.config['regularizationRateL1'], r.config['regularizationRateL2'])

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
    feeds[FeedLoc.rnn] = layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.rnn])), name='rnn_feed')
    feeds[FeedLoc.dense] = layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.dense])), name='dense_feed')
    feed_lens = [feeds[i].shape[-1] for i in range(FeedLoc.LEN)]

    # SYSTEM 1 : convolution
    # Make conv layers
    if feed_lens[FeedLoc.conv] > 0 and r.config['convType'].lower() != 'none':
        if r.config['convType'].lower() == 'filternet':
            # FilterNet style conv
            convLayers = []
            this_layer = feeds[FeedLoc.conv]
            for i in range(convCfg['layerCount']):
                convLayers.append(MakeLayerModule('conv', this_layer, k_reg=make_reg(reg_def), out_width=convCfg['filters'][i],
                    kernel_size=convCfg['kernelSz'][i], dilation=convCfg['dilation'][i],
                    dropout_rate=r.config['dropout'], batch_norm=r.config['batchNorm'],
                    name= f"conv1d_{i}_{convCfg['dilation'][i]}x"))
                if r.config['convCascade']:
                    # Convolutional networks feed into eachother, with skip connections
                    this_layer = convLayers[-1]

            if convCfg['layerCount'] == 0:
                this_layer = feeds[FeedLoc.conv]
            elif convCfg['layerCount'] == 1:
                this_layer = convLayers[0]
            elif convCfg['layerCount'] > 1:
                this_layer = layers.concatenate(convLayers, name="cat_conv")
        elif r.config['convType'].lower() == 'wavenet':
            # WaveNet style conv
            this_layer = MakeWaveNet(feeds[FeedLoc.conv], 
                stack_count = r.config['wnStackCount'],
                factor = r.config['wnFactor'],
                module_count = r.config['wnModuleCount'],
                out_width = r.config['wnWidth'],
                k_reg_def=reg_def)
        else:
            raise Exception("r.config['convType'] is unexpected")

        # Add LSTM feed
        if feed_lens[FeedLoc.rnn] > 0:
            this_layer = layers.concatenate([this_layer, feeds[FeedLoc.rnn]], name='cat_bottleneck')
    else:
        # No convolutional input
        this_layer = feeds[FeedLoc.rnn]

    # SYSTEM 2: RNN (LSTM/GRU)
    if r.config['rnnType'].lower() != 'none':
        # Bottleneck layer (to reduce size going to LSTM/GRU)
        bnw = r.config['bottleneckWidth']
        if bnw > 0:
            this_layer = MakeLayerModule('dense', this_layer, out_width=bnw, k_reg=make_reg(reg_def), dropout_rate=r.config['dropout'],
                batch_norm=r.config['batchNorm'], name= f"bottleneck_{bnw}")

        # LSTM/GRU layers
        rnn_type = r.config['rnnType'].lower()
        for i, neurons in enumerate(r.config['rnnWidths']):
            if neurons > 0:
                this_layer = MakeLayerModule(rnn_type, this_layer, out_width=neurons, k_reg=make_reg(reg_def), dropout_rate=r.config['dropout'],
                    batch_norm=r.config['batchNorm'], name= f'{rnn_type}_{i}_{neurons}')

    # SYSTEM 3: Dense/Fully connected
    # Add dense input feed
    if feed_lens[FeedLoc.dense] > 0:
        this_layer = layers.concatenate([this_layer, feeds[FeedLoc.dense]])

    # Add any dense layers
    for i, neurons in enumerate(r.config['denseWidths']):
        if neurons > 0:
            this_layer = MakeLayerModule('dense', this_layer, out_width=neurons, k_reg=make_reg(reg_def), dropout_rate=r.config['dropout'],
                batch_norm=r.config['batchNorm'], name= f'dense_{i}_{neurons}')

    # Output (dense) layer
    main_output = layers.Dense(units=r.outFeatureCount, activation='linear', name='final_output',
    kernel_regularizer=make_reg(reg_def)) (this_layer)
    
    r.model = CustomModel(inputs=feeds, outputs=[main_output])
    r.modelEpoch = -1
    r.trainHistory = {}

    # mape = mean absolute percentage error
    r.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error', 'mean_squared_error', \
        score_sq_any, dynamism])
    #r.model.build(input_shape=(None, r.inFeatureCount))

    return

#==========================================================================
def PrintNetwork(r):
    PrintConfigString(r.config)
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

    verbose = 1 if r.isBatch else 2
    
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

    r.neutralTrainAbsErr = np.mean(np.abs(trainY))
    r.neutralValAbsErr = np.mean(np.abs(valY))
    r.neutralTrainSqErr = np.mean(np.abs(trainY)**2)
    r.neutralValSqErr = np.mean(np.abs(valY)**2)

    #Callbacks
    callbacks = []

    fitnessCb = FitnessCb()
    fitnessCb.setup(trainX, trainY, valX, valY, r.neutralTrainSqErr, r.neutralValSqErr, r.neutralTrainAbsErr, r.neutralValAbsErr)
    callbacks.append(fitnessCb)

    checkpointCb = CheckpointCb()
    checkpointCb.setup(r.config['earlyStopping'])
    callbacks.append(checkpointCb)

    printoutCb = PrintoutCb()
    printoutCb.setup(r.config['epochs'])
    if verbose >= 2:
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
def TrainNetwork(r, inData, outData, final=True, plotMetrics=True):
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
 

    if hist.epoch and hist.epoch[-1]+1 != len(list(r.trainHistory.values())[0]):
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
    
    if plotMetrics:
        PlotTrainMetrics(r.trainHistory, plotWidth=r.config['plotWidth'])

    return

#==========================================================================
def MakeAndTrainNetwork(r, inData, outData):
    MakeNetwork(r)
    PrintNetwork(r)
    TrainNetwork(r, inData, outData)
    return

#==========================================================================
# Make several networks and choose the best
def MakeAndTrainPrunedNetwork(r, inData, outData, candidates = 5, trialEpochs = 16, drawPlots=True):

    # Create all models
    models = [0] * candidates
    trainHist = [0] * candidates
    for i in range(candidates):
        MakeNetwork(r)
        models[i] = r.model
        if i == 0 and drawPlots:
            PrintNetwork(r)
    printmd('**********************************************************************************')
    printmd('## PRUNED NETWORK')
    printmd(f'Training **{candidates}** candidate models for **{trialEpochs}** epochs, then selecting the best.')

    # Trial each model by a small amount of training
    epochBackup = r.config['epochs']
    r.config['epochs'] = trialEpochs
    fitnessTrain = np.zeros((candidates))
    fitnessVal = np.zeros((candidates))
    trainGradSum = np.zeros((candidates))
    valGradSum = np.zeros((candidates))
    for i in range(candidates):
        print('\n************************************')
        printmd(f'Training candidate model **{i}** out of {candidates}')
        r.model = models[i]
        r.modelEpoch = -1
        TrainNetwork(r, inData, outData, final=False, plotMetrics=False)
        trainHist[i] = r.trainHistory

        fitnessTrain[i] = r.trainHistory['fitness'][-1]
        fitnessVal[i] = r.trainHistory['val_fitness'][-1]
        # gradSum is a sum of the last 5 gradients (from the last 6 values)
        # of the log of the loss.
        # Should be negative. More negative = better
        # The most recent is weighted more than the first
        pastVal = min(6,trialEpochs)
        trainGradSum[i] = np.sum(np.diff(np.log(r.trainHistory['mean_squared_error'][-pastVal:])) * np.linspace(1,2,num=pastVal-1))
        valGradSum[i] = np.sum(np.diff(np.log(r.trainHistory['val_mean_squared_error'][-pastVal:])) * np.linspace(1,2,num=pastVal-1))
       
  
    # PICK THE BEST MODEL
    # Higher score = better
    rating = pd.DataFrame()
    rating['train'] = fitnessTrain / fitnessTrain.max() # 0 to 1 (1 being the best candidate)
    rating['val'] = fitnessVal / fitnessVal.max() # 0 to 1 (1 being the best candidate)
    # For gradient scores, 1 is the top rating, and it scales down from there
    # The amount that it drops is determined by 

    # Method prior to 2021-11-07:
    # trainGradScale = np.abs(np.mean(np.log(lossTrain)))
    # temp = trainGradSum / trainGradScale * 8
    # rating['trainGrad'] = np.clip(temp.min()-temp+1, -1, 1)

    rating['trainGrad'] = np.clip(-(trainGradSum / np.abs(trainGradSum.min())), -1, 1)
    rating['valGrad'] = np.clip(-(valGradSum / np.abs(valGradSum.min())), -1, 1)

    print('************************************')
    printmd('### All candidate model ratings:')

    # Weight each of the criteria
    rating['total'] = rating['train'] * 1 \
        + rating['val'] * 2 \
        + rating['trainGrad'] * 0.5 \
        + rating['valGrad'] * 0.5
    
    print(rating)
              
    bestI = rating['total'].argmax()
    printmd('**Chose candidate model: {}**'.format(bestI))
    printmd('**********************************************************************************')
    
    # Train on the best model
    r.config['epochs'] = epochBackup
    r.model = models[bestI]
    r.trainHistory = trainHist[bestI]
    TrainNetwork(r, inData, outData, plotMetrics=drawPlots)
    return
    
#==========================================================================
def TestNetwork(r, priceData, inData, outData, drawPlots=True):
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
        ax.set_title('Prices. Sample {} ({}) [{}]'.format(r.config['coinList'][sample], sample, r.batchRunName))
        ax.grid()
        
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
            ax.legend(handles = [l1, l2, l3, l4])
            ax.grid()
        # Save file if necessary
        if r.isBatch and sample == 0:
            try:
                directory = './batches/' + r.batchName
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + '/'+ r.batchRunName +'.png'
                plt.savefig(filename.replace(':','-'))
            except:
                print('\n\n SAVING FILE ERROR')
        plt.show()
    
    r.prediction = predictY
    # Plot prediction

    if drawPlots:
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
    val_y. With this function, y_pred is truncated to the length of val_y, cutting
    off the initial entries.
    The purpose here is that the start of the prediction is used for
    building state and not for evaluation. That section can overlap with 
    the training set as it's not used for evaluation.

    Updated note: Instead of this, I could perhaps just make a lambda layer that cuts off 
                  a number of timesteps. See https://stackoverflow.com/a/54750309

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



#==========================================================================
def MakeWaveNetModule(layer_input, out_width:int, kernel_size:int, dilation:int, name:str="", k_reg_def=None):
    # https://arxiv.org/pdf/1609.03499.pdf
    #        |-> [gate]   -|        |-> 1x1 conv -> skip output
    #        |             |-> (*) -|
    # input -|-> [filter] -|        |-> 1x1 conv -|
    #        |                                    |-> (+) -> dense output
    #        |------------------------------------|

    bias = False
    this_layer = layer_input
    conv_args = {
        'filters' : out_width, # filter count = number of outputs
        'kernel_size' : kernel_size, # size of all filters
        'dilation_rate' : dilation, # factor in how far back to look
        'use_bias' : bias, # Can experiment with this 
        'padding' : 'causal', # causal; don't look into the future
    }

    # tanh (filter)
    conv_args['name'] = f'{name}_conv_tanh'
    conv_args['activation'] = 'tanh'
    conv_args['kernel_regularizer'] = make_reg(k_reg_def)
    tanh = layers.Conv1D(**conv_args)(this_layer)

    # sigm (gate)
    conv_args['name'] = f'{name}_conv_sigm'
    conv_args['activation'] = 'sigmoid'
    conv_args['kernel_regularizer'] = make_reg(k_reg_def)
    sigm = layers.Conv1D(**conv_args)(this_layer)

    # multiply
    this_layer = layers.Multiply(name=f"{name}_mult")([tanh, sigm])

    # skip
    skip = layers.Dense(units=out_width, use_bias=bias, name=f"{name}_dense_skip", kernel_regularizer=make_reg(k_reg_def))(this_layer)

    # dense out
    this_layer = layers.Dense(units=out_width, use_bias=bias, name=f"{name}_dense_res", kernel_regularizer=make_reg(k_reg_def))(this_layer)
    this_layer = layers.Add(name=f"{name}_res_add")([this_layer, layer_input])

    return this_layer, skip

#==========================================================================
def MakeWaveNetStack(layer_input, factor:int, module_count:int, out_width:int, name:str="", k_reg_def=None):
    """Make a WaveNet stack, which is a cascading group of WaveNet modules. Each module
    has its dilation factor increased by factor

    Args:
        layer_input: [description]
        factor (int): the kernel_size and dilation factor (usually 2)
        module_count (int): Each module increases receptive filed by 'factor'. Total receptive field will be factor**modulecount
        out_width (int): filter count
        name ([type], optional): [description]. Defaults to None.

    Returns:
        this_layer, skips
    """
    this_layer = layer_input
    skips = []
    for i in range(module_count):
        this_layer, skip_new = MakeWaveNetModule(this_layer, out_width=out_width, 
        kernel_size=factor, dilation=factor**i, name=f"{name}_m{i}", k_reg_def=k_reg_def)
        skips.append(skip_new)
    
    return this_layer, skips

#==========================================================================
def MakeWaveNet(layer_input, module_count:int, out_width:int, name:str="", stack_count:int=1, factor:int=2, k_reg_def=None):
    # Starts with "Causal Conv"
    # "Causal Conv" is not well defined. I believe it's a 1x1 (aka Dense) to change the data width
    this_layer = layers.Dense(units=out_width, use_bias=False, name=f"{name}wn_cc",
    kernel_regularizer=make_reg(k_reg_def))(layer_input)

    # Main body of WaveNet:
    all_skips = []
    for s in range(stack_count):
        thisPre = f"{name}wn_s{s}" if stack_count > 1 else f"{name}wn"
        this_layer, skips = MakeWaveNetStack(this_layer,factor=factor, 
            module_count=module_count, out_width=out_width, name=thisPre, k_reg_def=k_reg_def)
        all_skips += skips

    receptive_field = stack_count * factor**module_count - (stack_count - 1)
    print(f"WaveNet receptive field = {receptive_field}h. {receptive_field/24} days.")

    # cap off a WaveNet network by summing the skip connections and adding relu
    this_layer = layers.Add(name=f"{name}wn_sum_skip")(all_skips)
    return layers.Activation('relu', name=f"{name}wn_act_relu")(this_layer)

    # Note that there should also be after this: 1x1, relu, 1x1, softmax
