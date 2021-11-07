# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:12:18 2017

@author: Dean
"""

import numpy as np
import matplotlib.pyplot as plt

import keras
from DataTypes import TrainMetrics
import pandas as pd
import keras.callbacks
import time
import os
from ClockworkRNN import CWRNN
from keras.callbacks import EarlyStopping

from scripts.DataTypes import FeedLoc
       
#==========================================================================
class ValidationCb(keras.callbacks.Callback):
    """
    This callback is used to validate/test the model after each epoch. Keras
    supports this functionality with validation_data in .fit(), but a downside
    with that for a LSTM is that it starts the validation without building
    state. This builds state before the prediction range - so the validation
    is a better representation of its actual usage. Allows for better-informed
    early stopping.
    """
    def __init__(self):
        pass
    
    
    def setup(self, xData, yTarget, trainMetrics, patience):
        self.xData = xData # DOCUMENT THESE !@#$
        self.yTarget = np.array(yTarget) # Target of predictions
        self.targetSize = yTarget.size # Number of points
        self.targetTimeSteps = yTarget.shape[-2]
        
        # Determine the thresholds for penalising lack of movement
        # This is to avoid uneventful results that don't do anything
        # Do this by comparing to a smoothed version of the target
        totalDiff = 0
        diffCount = 0
        windowSz = 10
        window = np.ones((windowSz,))/windowSz
        for sample in range(yTarget.shape[0]):
            for outTarget in range(yTarget.shape[-1]):
                smoothed = np.convolve(yTarget[sample,:,outTarget], window, mode='valid')
                totalDiff += np.sum(np.abs(np.diff(smoothed)))
                diffCount += yTarget[sample,:,outTarget].size-1
        avgDiffTarget = totalDiff / diffCount
        self.avgDiffUpper = avgDiffTarget / 4 # above this, there's no penalisation
        self.avgDiffLower = avgDiffTarget / 10 # Below this, the
        # penalty is a maximum
        
        self.trainMetrics = trainMetrics # Links to the r.trainMetrics
        # Tracking the best result:
        self.bestFitness = 0
        self.bestWeights = []
        self.bestEpoch = 0
        # Early stopping
        self.wait = 0
        self.patience = patience # Epochs before stopping early
        # Set to 0 to disable early stopping
        self.prevFitness = 0
        self.stopped_epoch = 0
           
    def on_epoch_end(self, epoch, logs={}):
        self.trainMetrics.curEpoch += 1
        thisEpoch = self.trainMetrics.curEpoch
        predictY = self.model.predict(self.xData, batch_size=100)
        err = predictY[:,-self.targetTimeSteps:,:] - self.yTarget
        
        val_abs_err = np.sum(np.abs(err)) / self.targetSize
        val_sq_err = np.sum(err**2) / self.targetSize
        
               
        fitness = 1/val_sq_err
        # Penalise predictions don't move enough
        thisDiff = np.mean(np.abs(np.diff(predictY, axis=1)))
        debug = 0
        if debug: print('Dif Score {:5f}, Lower {:5f}, Upper {:5f}'.format(thisDiff, self.avgDiffLower, self.avgDiffUpper), end='')
        penalty = 1.
        if thisDiff < self.avgDiffUpper:
            penaltyLower = 0.001
            penaltyUpper = 1
            pos = (thisDiff - self.avgDiffLower) / (self.avgDiffUpper - self.avgDiffLower) # 0 to 1
            penalty = (pos + penaltyLower) * (penaltyUpper - penaltyLower)
            penalty = np.clip(penalty, penaltyLower, penaltyUpper)
            fitness *= penalty
            if debug: print(' scaler = {:5f}'.format(penalty))
        if debug: print('') # new line
        
        logs['fitness'] = fitness # for choosing the best model
        
        if fitness > self.bestFitness:
            self.bestFitness = fitness
            self.bestWeights = self.model.get_weights()
            self.bestEpoch = thisEpoch
            print('Epoch {:2} - Fitness= {:7.5f} - DiffScaler= {:5.3f} - New top!'.format(thisEpoch, fitness, penalty))
        elif (thisEpoch%10)==0:
            print('Epoch {:2} - Fitness= {:7.3f} - DiffScaler= {:5.3f}'.format(thisEpoch, fitness, penalty))
        
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
                    print('No improvement in {} epochs'.format(self.patience))
                    print('STOPPING TRAINING AT EPOCH {}'.format(thisEpoch))
        
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
    #PLot Training
    if not subplot:
        plt.figure()
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
        plt.suptitle('Training Errors')
        plt.show()
    
    return (maxY, minY)

def PrepConvConfig(r):
    convConf = dict()
    convConf['convDilation'] = r.config['convDilation']
    convConf['convFilters']  = r.config['convFilters']  
    convConf['convKernelSz'] = r.config['convKernelSz']

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
def MakeNetwork(r):
    # Prep convolution config
    convConf = PrepConvConfig(r)

    #Make a Neural Network
    if type(r.kerasOpt) == int:
        # beta_1 = exponential decay rate for 1st moment estimates. Default=0.9
        # beta_2 = exponential decay rate for 2nd moment estimates. Default=0.999
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9)
        r.optimiser = 'Adam'
    else:
        opt = r.kerasOpt
        r.optimiser = r.kerasOptStr

    feeds = [[] for i in range(FeedLoc.LEN)]
    
    # Keras functional API
    # Input
    feeds[FeedLoc.conv] = keras.layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.conv])), name='conv_feed')

    # Make conv layers
    convLayers = []
    for i in range(convConf['layerCount']):
        convLayers.append(keras.layers.Conv1D(
            filters=convConf['convFilters'][i], 
            kernel_size=convConf['convKernelSz'][i],
            dilation_rate=convConf['convDilation'][i],
            # input_shape=(None, r.inFeatureCount),
            use_bias=True, padding='causal',
            name=f"conv1d_{i}_{convConf['convDilation'][i]}x")(feeds[FeedLoc.conv]))

    if convConf['layerCount'] == 0:
        conv_out = feeds[FeedLoc.conv]
    elif convConf['layerCount'] == 1:
        conv_out = convLayers[0]
    elif convConf['layerCount'] > 1:
        conv_out = keras.layers.concatenate(convLayers)



    # Make the LSTM layers
    # LSTM input
    feeds[FeedLoc.lstm] = keras.layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.lstm])), name='lstm_feed')
    lstm_input = keras.layers.concatenate([conv_out, feeds[FeedLoc.lstm]])

    lstmLayerCount = len(r.config['neurons'])
    
    for i, neurons in enumerate(r.config['neurons']):
        is_first_layer = (i == 0)
        is_last_layer = (i == lstmLayerCount - 1)

        lstm_args = {
            'units' : neurons,
            'activation' : 'tanh',
            'stateful' : False,
            'return_sequences' : True, # I'm including output values for all time steps, so always true
            'dropout' : r.config['dropout']
        }
        # if is_first_layer and convLayerCount == 0:
        #     lstm_args['input_shape'] = (None, r.inFeatureCount)
        this_input = lstm_input if is_first_layer else lstm_out
        lstm_out = keras.layers.LSTM(**lstm_args)(this_input)
    
    # Dense layers
    # Dense input
    feeds[FeedLoc.dense] = keras.layers.Input(shape=(None, np.sum(r.feedLocFeatures[FeedLoc.dense])), name='dense_feed')
    dense_input = keras.layers.concatenate([lstm_out, feeds[FeedLoc.dense]])

    main_output = keras.layers.Dense(r.outFeatureCount, name='final_output')(dense_input)
    r.model = keras.models.Model(inputs=feeds, outputs=[main_output])

    # mape = mean absolute percentage error
    r.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
    #r.model.build(input_shape=(None, r.inFeatureCount))
    
    r.trainMetrics = TrainMetrics()

    return

#==========================================================================
def PrintNetwork(r):
    r.model.summary()

    keras.utils.plot_model(r.model, show_shapes=True, to_file='model.png')

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
    validationCb.setup(valInData, outData[:,valI,:], r.trainMetrics, r.config['earlyStopping'])
        
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
    print('\nStarting training. Max {} epochs'.format(epochsLeft))
    
    trainX = [arr[:,r.tInd['train'],:] for arr in inData]
    trainY = outData[:, r.tInd['train']]
    valY = outData[:, r.tInd['val']]
    
    r.neutralTrainAbsErr = np.sum(np.abs(trainY)) / trainY.size
    r.neutralValAbsErr = np.sum(np.abs(valY)) / valY.size
    r.neutralTrainSqErr = np.sum(np.abs(trainY)**2) / trainY.size
    r.neutralValSqErr = np.sum(np.abs(valY)**2) / valY.size
    
    start = time.time()

    hist = r.model.fit(trainX, trainY, epochs=epochsLeft, 
                     batch_size=r.sampleCount, shuffle=True,
                     verbose=0, callbacks=callbacks)
    
    end = time.time()
    r.trainTime = end-start
    print('Training Time: {0}'.format(r.trainTime))
    PlotTrainMetrics(r)
    
    if final and r.config['revertToBest']:
        if validationCb.bestEpoch > 0:
            print('Reverting to the model with best validation (epoch {})'.format(validationCb.bestEpoch))
    #        r.model.load_weights(fileBestWeights)
            r.model.set_weights(validationCb.bestWeights)

    return


#==========================================================================
def MakeAndTrainNetwork(r, inData, outData):
    MakeNetwork(r)
    TrainNetwork(r, inData, outData)
    return

#==========================================================================
# Make several networks and choose the best
def MakeAndTrainPrunedNetwork(r, inData, outData):
    # SETTINGS
    candidates = 5
    trialEpochs = 10

    # Create all models
    models = [0] * candidates
    trainMetrics = [0] * candidates
    for i in range(candidates):
        MakeNetwork(r)
        models[i] = r.model
        trainMetrics[i] = r.trainMetrics
        if i == 0:
            PrintNetwork(r)
    print('\n**********************************************************************************')
    print('PRUNED NETWORK')
    print(f'Training {candidates} candidate networks for {trialEpochs} epochs, then selecting the best.')

    # Trial each model by a small amount of training
    epochBackup = r.config['epochs']
    r.config['epochs'] = trialEpochs
    lossTrain = np.zeros((candidates))
    lossVal = np.zeros((candidates))
    trainGradSum = np.zeros((candidates))
    valGradSum = np.zeros((candidates))
    fitness = np.zeros((candidates))
    for i in range(candidates):
        print('\n**********************************************************************************')
        print('Training model {} out of {}'.format(i, candidates))
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
    print('All candidate scores:')
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
    print('Chosen model: {}'.format(bestI))
    
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
        fig, axs = plt.subplots(plotsHigh, 1, sharex=True, figsize=(12,3*plotsHigh)) # TODO get this working
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
