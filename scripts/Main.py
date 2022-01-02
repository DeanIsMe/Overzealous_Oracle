# -*- coding: utf-8 -*-
"""
Main module for running and testing neural networks for ASX analysis
Created on Thu Oct 12 20:17:14 2017

@author: Dean
"""

import numpy as np
from FeatureExtraction import CalcFavScores
from Config import GetConfig
from NeuralNet import SplitData, MakeAndTrainNetwork, TestNetwork
from TestSequences import GetInSeq
import InputData
import matplotlib.pyplot as plt
import time

def PrintDataLimits(inData, outData):
    print("\r\ninData MAX:")
    print(np.max(inData, axis=1))
    print("\r\ninData MIN:")
    print(np.min(inData, axis=1))
    
    print("\r\noutData MAX:")
    print(np.max(outData, axis=1))
    print("\r\noutData MIN:")
    print(np.min(outData, axis=1))

config = GetConfig()

#At this point, the stock data should have all gaps filled in, and be inflation
#adjusted. Shape should be (Stocks, Timesteps, Features)
    
inSeq = GetTestInSeq()
outData = CalcFavScores(config, inSeq[:,:])

inFeatures = 1
samples = inSeq.shape[0]
timesteps = inSeq.shape[1]
inData = np.zeros((samples, timesteps, inFeatures))

# Single Run
##Set the time sequence
##22/10/2017: a normalised input sequence was shown to be best
##inData[:,:,0] = InputData.LogDiffSeries(inSeq)
##inData[:,:,0] = InputData.LogSeries(inSeq)
inData[:,:,0] = InputData.NormaliseEachSeq(inSeq)
##inData[:,:,0] = InputData.inSeq
#
#Scale values to a reasonable range
#17/12/2017: dividing by 90th percentile was found to be a good scale
inData = inData / np.percentile(inData, 90) * 1
outData = outData / np.percentile(outData, 90) * 1

PrintDataLimits(inData, outData)

(trainX, trainY, valX, valY, testX, testY, tInd) = SplitData(config, inData, outData)
model, trainTime = MakeAndTrainNetwork(config, trainX, trainY, valX, valY)
TestNetwork(config, model, inSeq, inData, outData, tInd)

# Batch Run
#scaler = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
#runs2= len(scaler)
#models2 = [0]*runs2
#trainD2 = [0]*runs2
#testErr2 = [0]*runs2
#trainTime2 = [0]*runs2
#for run2 in range(runs2):
#    print('\n\nRUN2 {0}'.format(run2))
#    inScale = scaler[run2]
#    
#    runs = len(scaler)
#    models = [0]*runs
#    trainD = [0]*runs
#    testErr = [0]*runs
#    trainTime = [0]*runs
#    for run in range(runs):
#        print('\n\nRUN2 {0}'.format(run2))
#        print('\n\nRUN {0}'.format(run))
#        outScale = scaler[run]
#        
#        thisOutData = outData / np.percentile(outData, 90) * outScale
#        thisInData = inData / np.percentile(inData, 90) * inScale
#            
#        (trainX, trainY, valX, valY, testX, testY, tInd) = SplitData(config, thisInData, thisOutData)
#        models[run], trainD[run], trainTime[run] = MakeAndTrainNetwork(config, trainX, trainY, valX, valY)
#        testErr[run] = TestNetwork(config, models[run], inSeq, thisInData, thisOutData, tInd)
#    
#    models2[run2] = models
#    trainD2[run2] = trainD
#    testErr2[run2] = testErr
#    trainTime2[run2] = trainTime
#
#namePrefix = 'IOSmallScale'
#np.save(namePrefix + 'TrainMetrics', trainD2)
#np.save(namePrefix + 'Err', testErr2)
#np.save(namePrefix + 'TrainTime', trainTime2)
