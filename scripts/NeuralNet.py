# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:12:18 2017

@author: Dean
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model

import keras
from DataTypes import TrainData
import pandas as pd
import keras.callbacks
import time
import os
from ClockworkRNN import CWRNN
from keras.callbacks import EarlyStopping
       
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
    

            
    def setup(self, xData, yTarget, trainData, patience):
        self.xData = np.array(xData) # DOCUMENT THESE !@#$
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
  
        self.trainData = trainData # Links to the r.trainData
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
        self.trainData.curEpoch += 1
        thisEpoch = self.trainData.curEpoch
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
        
        self.trainData.absErrVal.append(val_abs_err)
        self.trainData.lossVal.append(val_sq_err)
        self.trainData.lossTrain.append(logs['loss'])
        self.trainData.absErrTrain.append(logs['mean_absolute_error'])
        self.trainData.fitness.append(fitness)
        
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
def PlotTrainData(r, subplot=False):
    #PLot Training
    if not subplot:
        plt.figure()
#    ax = plt.gca()
    maxY = -9e9
    minY = 9e9
    
    lines = []
    lines.append({'label':'TrainSq',
                  'data':r.neutralTrainSqErr / r.trainData.lossTrain,
                  'ls':'-', 'color':'C0'})
    lines.append({'label':'ValSq',
                  'data':r.neutralValSqErr / r.trainData.lossVal,
                  'ls':'-', 'color':'C1'})
    lines.append({'label':'TrainAbs',
                  'data':r.neutralTrainAbsErr / r.trainData.absErrTrain,
                  'ls':':', 'color':'C0'})
    lines.append({'label':'ValAbs',
                  'data':r.neutralValAbsErr / r.trainData.absErrVal,
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
        plt.suptitle('Training Scores')
        plt.show()
    
    return (maxY, minY)

#==========================================================================
def _MakeNetwork(r):
    #Make a Neural Network
    
    if type(r.kerasOpt) == int:
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9)
        r.optimiser = 'Adam()'
    else:
        opt = r.kerasOpt
        r.optimiser = 'kerasOpt'
    r.kerasOptStr = str(opt)
    
    r.model = Sequential()
    # Add the LSTM layers
    numLayers = len(r.config['neurons'])
    for i, neurons in enumerate(r.config['neurons']):
        if i == numLayers-1:
            act = 'tanh'
        else:
            act = 'tanh'
        
        if i == 0:
            r.model.add(keras.layers.LSTM(neurons, activation=act, 
                            batch_input_shape=(r.sampleCount, None, r.inFeatureCount),
                             stateful=False, return_sequences=True,
                             dropout=r.config['dropout']))
        else:
            r.model.add(keras.layers.LSTM(neurons, activation=act,
                             stateful=False, return_sequences=True,
                             dropout=r.config['dropout']))

    
    # if bias is allowed in the output layer, then the output ends up with a
    # 'floor' from which it rises, which is inapproriate for this type of output
    # 24/03/2018
    r.model.add(keras.layers.Dense(r.outFeatureCount, use_bias=False))
        
# Branched network (linear + saturating branches) 18/02/2018
#    main_input = Input(batch_shape=(samples, 1, inFeatures), name='main_input')
#    lstm1 = LSTM(64, name='LSTM1')(main_input)
#    # Saturating branch, for binary decisions
#    denseSat1 = Dense(64, activation='tanh', name='DenseSat1')(lstm1)
#    # Linear branch, for passing values
#    denseLin1 = Dense(64, activation='linear', name='DenseLin1')(lstm1)
#    # Merge the branches together
#    merge = keras.layers.concatenate([denseSat1, denseLin1])
#    # Final output layer
#    main_output = Dense(outputDim, name='Output')(merge)
#    r.model = Model(inputs=[main_input], outputs=[main_output])
    
    # mape = mean absolute percentage error
    r.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
    r.modelSummary = r.model.summary()
    print(r.modelSummary)
    
    r.trainData = TrainData()
    
    return


#==========================================================================
def _TrainNetwork(r, inData, outData, final=True):
    """
    final == True indicates that this is the final call for TrainNetwork for
    this model.
    """
    tInd = _CalcIndices(inData.shape[-2], r.config['dataRatios'], r.config['excludeRecentDays'])
    
    #Callbacks
    callbacks = []
    
    # Callback to validate data
    validationCb = ValidationCb()
    valI = tInd['val'] # Validation indices
    startPredict = max(0, valI.min()-100) # This number of time steps are used to build state before starting predictions
    validationCb.setup(inData[:,startPredict:valI.max()+1,:], outData[:,valI,:], r.trainData, r.config['earlyStopping'])
        
    epochsLeft = r.config['epochs'] - r.trainData.curEpoch
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
    print('\nStarting training. Max {} epochs'.format(epochsLeft));
    trainX = inData[:, tInd['train']]
    trainY = outData[:, tInd['train']]
    valY = outData[:, tInd['val']]
    
    r.neutralTrainAbsErr = np.sum(np.abs(trainY)) / trainY.size
    r.neutralValAbsErr = np.sum(np.abs(valY)) / valY.size
    r.neutralTrainSqErr = np.sum(np.abs(trainY)**2) / trainY.size
    r.neutralValSqErr = np.sum(np.abs(valY)**2) / valY.size
    
    start = time.time()
    (samples, trainSteps, inFeatures) = trainX.shape

    hist = r.model.fit(trainX, trainY, epochs=epochsLeft, 
                     batch_size=samples, shuffle=True,
                     verbose=0, callbacks=callbacks)
    
    end = time.time()
    r.trainTime = end-start
    print('Training Time: {0}'.format(r.trainTime))
    PlotTrainData(r)
    
    if final and r.config['revertToBest']:
        if validationCb.bestEpoch > 0:
            print('Reverting to the model with best validation (epoch {})'.format(validationCb.bestEpoch))
    #        r.model.load_weights(fileBestWeights)
            r.model.set_weights(validationCb.bestWeights)

    return


#==========================================================================
def MakeAndTrainNetwork(r, inData, outData):
    _MakeNetwork(r)
    _TrainNetwork(r, inData, outData)
    return

#==========================================================================
# Make several networks and choose the best
def MakeAndTrainPrunedNetwork(r, inData, outData):
    # Create all models
    options = 5
    testEpochs = 10
    print('\nPRUNED NETWORK. Making {} networks for {} epochs.'.format(options, testEpochs))
    models = [0] * options
    trainData = [0] * options
    for i in range(options):
        _MakeNetwork(r)
        models[i] = r.model
        trainData[i] = r.trainData
    # Perform initial training to test each model
    epochBackup = r.config['epochs']
    r.config['epochs'] = testEpochs
    lossTrain = np.zeros((options))
    lossVal = np.zeros((options))
    trainGradSum = np.zeros((options))
    valGradSum = np.zeros((options))
    fitness = np.zeros((options))
    for i in range(options):
        print('Training model {} out of {}'.format(i, options))
        r.model = models[i]
        r.trainData = trainData[i]
        _TrainNetwork(r, inData, outData, final=False)
        
        lossTrain[i] = r.trainData.lossTrain[-1]
        lossVal[i] = r.trainData.lossVal[-1]
        fitness[i] = np.max(r.trainData.fitness)
        # gradSum is a sum of the last 5 gradients (from the last 6 values)
        # of the log of the loss.
        # Should be negative. More negative = better
        # The most recent is weighted more than the first
        pastVal = min(6,testEpochs)
        trainGradSum[i] = np.sum(np.diff(np.log(r.trainData.lossTrain[-pastVal:])) * np.linspace(1,2,num=pastVal-1))
        valGradSum[i] = np.sum(np.diff(np.log(r.trainData.lossVal[-pastVal:])) * np.linspace(1,2,num=pastVal-1))
       
  
    # PICK THE BEST MODEL
    # Higher score = better
    scores = pd.DataFrame()
    scores['train'] = lossTrain.min()/lossTrain
    scores['val'] = fitness/fitness.max()
    # For gradient scores, 1 is the top score, and it scales down from there
    # The amount that it drops is determined by 
    trainGradScale = np.abs(np.mean(np.log(lossTrain)))
    temp = trainGradSum / trainGradScale * 8
    scores['trainGrad'] = np.clip(temp.min()-temp+1, -1, 1)
    
    valGradScale = np.abs(np.mean(np.log(lossVal)))
    temp = valGradSum / valGradScale * 8
    scores['valGrad'] = np.clip(temp.min()-temp+1, -1, 1)
        
    print('All Scores:')
    print(scores)
    
    # Weight each of the scores
    scores['train'] *= 1
    scores['val'] *= 2
    scores['trainGrad'] *= 0.1
    scores['valGrad'] *= 1
    
    totalScore = scores.sum(axis=1)
              
    bestI = totalScore.argmax()
     
    print('Total:')
    print(totalScore)
    print('Chosen model: {}'.format(bestI))
    
    # Train on the best model
    r.config['epochs'] = epochBackup
    r.model = models[bestI]
    r.trainData = trainData[bestI]
    _TrainNetwork(r, inData, outData)
    return
    
#==========================================================================
def TestNetwork(r, dailyPrice, inData, outData):
    tInd = _CalcIndices(inData.shape[-2], r.config['dataRatios'], r.config['excludeRecentDays'])
    tPlot = np.r_[0:inData.shape[-2]] # Range of output plot (all data)
    samples = inData.shape[-3]
    if (r.config['dataRatios'][2] > 0.1):
        print('WARNING! TestNetwork uses Val Data as the test data, but Test Data also exists. ')
        print(r.config['dataRatios'])
    testI = tInd['val'] # Validation indices used as test
    
    #Predictions
    predictY = r.model.predict(inData, batch_size=samples)
    
    def _PlotOutput(dailyPrice, out, predict, tRange, sample):
        """Plot a single output feature of 1 sample"""
        plotsHigh = 1+r.outFeatureCount
        plt.figure(1, figsize=(15,4*plotsHigh))
        plt.subplot(plotsHigh, 1, 1)
        plt.plot(dailyPrice[sample, tRange]) # Daily data
        plt.title('Prices. Sample {} ({}) [{}]'.format(r.coinList[sample], sample, r.batchRunName))
        
        for feature in range(r.outFeatureCount):
            plt.subplot(plotsHigh, 1, 2+feature)
            predictYPlot = predictY[sample, :, feature]
            outPlot = out[sample, tRange, feature]
            l1, = plt.plot(tRange, outPlot, label='Actual')
            l2, = plt.plot(tRange, predictYPlot, label='Prediction')
            l3, = plt.plot([tInd['train'][0], tInd['train'][0]], [np.min(outPlot), np.max(outPlot)], label='TrainStart')
            l4, = plt.plot([tInd['train'][-1], tInd['train'][-1]], [np.min(outPlot), np.max(outPlot)], label='TrainEnd')
            l0, = plt.plot([tRange[0], tRange[-1]], [0, 0])
            plt.title('Output Feature {} ({}-{}steps)'.format(feature, r.config['outputRanges'][feature][0], r.config['outputRanges'][feature][1]))
            plt.legend(handles = [l1, l2, l3 , l4])
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
    for s in range(samples):
        _PlotOutput(dailyPrice, outData, predictY, tPlot, s)
    
    r.testAbsErr = np.sum(np.abs(predictY[:,testI,:] - outData[:,testI,:])) / predictY[:,testI,:].size
    r.neutralTestAbsErr = np.sum(np.abs(outData[:,testI,:])) / outData[:,testI,:].size
    r.testScore = r.neutralTestAbsErr / r.testAbsErr
    
    r.trainAbsErr = np.sum(np.abs(predictY[:,tInd['train'],:] - outData[:,tInd['train'],:])) / predictY[:,tInd['train'],:].size
    r.trainScore = r.neutralTrainAbsErr / r.trainAbsErr
    
    # Assess the level of movement (some networks don't train and the result
    # is just a straight line)
    
    
    # Assess whether or not a 'floor' is occurring - if a large percent of the
    # data is close to the minimum
    
    
    print('Train Score: {:5}\nTest Score: {:5} (1=neutral)'.format(r.trainScore, r.testScore))
    return
