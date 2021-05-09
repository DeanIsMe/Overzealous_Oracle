# -*- coding: utf-8 -*-
"""
Main - Cryptocurrency analysis
Created on Dec 17  2017
@author: Dean
"""

#%% 
# Set the seed for repeatable results (careful with this use)
#print('FIXING SEED FOR REPEATABLE RESULTS')
#from numpy.random import seed
#seed(5)
#from tensorflow import set_random_seed
#set_random_seed(3)

%matplotlib widget

import numpy as np
import keras
import FeatureExtraction as FE
from Config_CC import GetConfig
from NeuralNet import MakeAndTrainNetwork, TestNetwork, MakeAndTrainPrunedNetwork
from DataTypes import ModelResult
from TestSequences import GetInSeq
import InputData as indata
import copy
from keras import backend as KB
import datetime

import matplotlib.pyplot as plt
import time
import Crypto_GetData as cgd
import pickle
from datetime import datetime

from keras import regularizers

#D:/Dean/Uni/Shares/Python Keras Tensorflow
def PrintDataLimits(inData, outData):
    print('Input Columns:')
    print(r.inDataColumns)
    print("\r\ninData 90th Percentile:")
    print(np.percentile(inData, 90, axis=1))
    print("\r\ninData 10th Percentile:")
    print(np.percentile(inData, 10, axis=1))
    
    print("\r\noutData 90th Percentile:")
    print(np.percentile(outData, 90, axis=1))
    print("\r\noutData 10th Percentile:")
    print(np.percentile(outData, 10, axis=1))

    
#def FillResultSetup(r):
#    # Fills basic information into the result class r
#    # r is ModelResult()
#    r.samples = samples
#    r.timesteps  = timesteps
#    r.inFeatures = inFeatures
#    r.outFeatures = outFeatures
#    r.config = copy.deepcopy(config)
#    r.coinList = coinList
#    r.inDataColumns = dfs[0].columns


KB.clear_session()

r = ModelResult()  
r.config = GetConfig() 

#At this point, the stock data should have all gaps filled in
#Shape should be (Stocks, Timesteps, Features
#Columns should be ['close', 'high', 'low', 'volumeto']

#r.coinList = ['ETH','BTC','BCH','XRP','LTC','XLM','NEO','EOS','XEM', 'IOT','DOGE','ADA','POT','VET','XLM','ETC']
#r.coinList = ['ETH','BTC','BCH','XRP','LTC']
r.coinList = ['ETH', 'ETC']
#r.coinList = ['ETH']

# %% Get Data

if 0:
    numHours = 24*180
    dfs = cgd.GetHourlyDf(r.coinList, numHours) # a list of data frames
    # To save a data set:
    date_str = datetime.now().strftime('%Y-%m-%d')
    filehandler = open(f'../indata/dfs_{len(dfs)}coins_{numHours}hours_{date_str}.pickle', 'wb')
    pickle.dump(dfs, filehandler)
    filehandler.close()
else:
    # !@#$
#    filehandler = open('dfs_5coins_40days_2018-02-17.pickle', 'rb')
    filehandler = open('../indata/dfs_2coins_4320hours_2021-05-09.pickle', 'rb')
    dfs = pickle.load(filehandler)
    filehandler.close()

# %% Prep the training data
r.samples = len(dfs)
r.timesteps = dfs[0].shape[-2]

prices = np.zeros((r.samples, r.timesteps))
for i in np.arange(r.samples):
    prices[i, :] =  np.array(dfs[i]['close'])

FE.AddLogDiff(r, dfs)
FE.AddVix(r, dfs, prices)
FE.AddRsi(r, dfs)
FE.AddEma(r, dfs)
FE.ScaleLoadedData(dfs) # High, Low, etc

r.inDataColumns = list(dfs[0].columns)
r.inFeatures = dfs[0].shape[-1]

# Plot a small sample of the input data
FE.PlotInData(r, dfs, 0, [0, 50])

# Convert to a numpy array
inData = np.zeros((r.samples, r.timesteps, r.inFeatures))
for i, df in enumerate(dfs):
    inData[i] = np.array(df)

# OUTPUT DATA
outData = FE.CalcFavScores(r.config, prices)
r.outFeatures = outData.shape[-1]

#Scale output values to a reasonable range
#17/12/2017: dividing by 90th percentile was found to be a good scale for SGD
for i in np.arange(r.outFeatures):
    outData[:,:,i] /= np.percentile(np.abs(outData[:,:,i]), 90)
    
# Print out data
FE.PlotOutData(r, prices, outData, 0)



 # %%
 single = True
if single:
# *****************************************************************************
# Single Run
    r.isBatch = False
    r.batchRunName = ''
    
    r.config['earlyStopping'] = 0
    r.config['neurons'] = [512, 256, 128]
    r.config['epochs'] = 64
    r.config['inScale'] = 1
    r.config['outScale'] = 1
    r.config['revertToBest'] = True
    
    thisInData = inData * r.config['inScale']
    thisOutData = outData * r.config['outScale']
    PrintDataLimits(thisInData, thisOutData)
    MakeAndTrainNetwork(r, thisInData, thisOutData)
#    MakeAndTrainPrunedNetwork(r, thisInData, thisOutData)
    TestNetwork(r, prices, thisInData, thisOutData)

else:
    # *****************************************************************************
    # Batch Run
    #
    
    

#    bat1Name = 'InScale'
#    bat1Val = [0.1, 1, 10, 100]
#    
#    bat2Name = 'OutScale'
#    bat2Val = [0.01, 0.1, 1, 10]
    
    bat1Name = 'InScale'
    bat1Val = [1, 100]
    
    bat2Name = 'OutScale'
    bat2Val = [0.1, 1]

    
    bat1Len = len(bat1Val)
    bat2Len = len(bat2Val)
    
    results = [0]*bat2Len
    r.isBatch = True
    r.batchName = str(datetime.date.today()) + '_' + \
    str(datetime.datetime.now().hour) + '_' + bat1Name + '_' + bat2Name
    startR = r
    
    for idx2, val2 in enumerate(bat2Val):
        results[idx2] = [0]*bat1Len
        # Change for batch2
        
        for idx1, val1 in enumerate(bat1Val):
            KB.clear_session()
            results[idx2][idx1] = copy.deepcopy(startR)
            r = results[idx2][idx1]
            
            print('\n\nBATCH RUN ({}, {})'.format(idx2, idx1))
            r.batchRunName = '{}:{}, {}:{}'.format(bat2Name, val2, bat1Name, val1)
            print(r.batchRunName)
             
            r.config['earlyStopping'] = 20
            # *********************************************************************
            # Change for this batch
#            r.config['epochs'] = 1
            r.config['dropout'] = 0
            r.config['inScale'] = val1
            r.config['outScale'] = val2
            
            
#            r.config['outputRanges'] = list(np.ceil(np.array([[12,48]]) * val2).astype(int))
#            fScale = val1 * val2
#            r.config['vixMaxPeriodPast'] = np.ceil(40 * 5 * fScale).astype(int)
#            r.config['rsiWindowLen'] = np.ceil(14 * 5 * fScale).astype(int) # The span of the EMA calc for RSI
#            r.config['emaLengths'] = list(np.ceil(np.array([9, 12, 26]) * 5 * fScale).astype(int)) # The span of the EMAs
       
##            filehandler = open('dfs_10coins_180days_2018_04_20.pickle', 'rb')
#            filehandler = open('dfs_5coins_40days_2018-02-17.pickle', 'rb')
#            dfs = pickle.load(filehandler)
#            filehandler.close()
#            
##            r.coinList = r.coinList[:val2]
##            dfs = dfs[:val2]
##            trainPoints = val1
#            
#            testPoints = 500
#
#            # Keep the same number of testing points each time
#            tPoints = np.array([trainPoints, testPoints, 100])
#            r.config['dataRatios'] = tPoints / tPoints.sum()
#            # Keep the test data as the same each time
#            for i in range(len(dfs)):
#                dfs[i] = dfs[i][-tPoints.sum():] # cut out the start       
#            
#            # STANDARD PROCESSING
#            r.samples = len(dfs)
#            r.timesteps = dfs[0].shape[-2]
#
#            prices = np.zeros((r.samples, r.timesteps))
#            for i in np.arange(r.samples):
#                prices[i, :] =  np.array(dfs[i]['close'])
#            
#            FE.AddLogDiff(r, dfs)
#            FE.AddVix(r, dfs, prices)
#            FE.AddRsi(r, dfs)
#            FE.AddEma(r, dfs)
#            FE.ScaleLoadedData(dfs) # High, Low, etc
#            
#            r.inDataColumns = list(dfs[0].columns)
#            r.inFeatures = dfs[0].shape[-1]
#            
#            # Convert to a numpy array
#            inData = np.zeros((r.samples, r.timesteps, r.inFeatures));
#            for i, df in enumerate(dfs):
#                inData[i] = np.array(df)
#            	
#            # OUTPUT DATA
#            outData = FE.CalcFavScores(r.config, prices)
#            r.outFeatures = outData.shape[-1]
#            
#            #Scale output values to a reasonable range
#            #17/12/2017: dividing by 90th percentile was found to be a good scale for SGD
#            for i in np.arange(r.outFeatures):
#                outData[:,:,i] /= np.percentile(np.abs(outData[:,:,i]), 90)
            # *********************************************************************
            
            thisInData = inData * r.config['inScale']
            thisOutData = outData * r.config['outScale']
#            PrintDataLimits(thisInData, thisOutData)
            MakeAndTrainNetwork(r, thisInData, thisOutData)
#            MakeAndTrainPrunedNetwork(r, thisInData, thisOutData)
            TestNetwork(r, prices, thisInData, thisOutData)
    
    print('\n\nBATCH RUN FINISHED!\n')
    # SAVE THE DATA
    # Clear the model so it can pickle
    models = [0] * bat2Len
    for idx2, rList in enumerate(results):
        models[idx2] = [0]*bat1Len
        for idx1, r in enumerate(rList):
            r = results[idx2][idx1]
            models[idx2][idx1] = r.model
            r.model = 0
            r.kerasOpt = 0
    
    filename = r.batchName + '.pickle'
    filehandler = open(filename, 'wb') 
    pickle.dump(results, filehandler)
    filehandler.close()
    
    # Copy the model back in
    for idx2, rList in enumerate(results):
        for idx1, r in enumerate(rList):
            results[idx2][idx1].model = models[idx2][idx1]
    
    #Go to sleep
    import os
    print('Going to sleep...')
    os.startfile ('C:\\Users\\Dean\\Desktop\\Sleep.lnk')