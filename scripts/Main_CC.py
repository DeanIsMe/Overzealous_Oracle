# -*- coding: utf-8 -*-
"""
Main - Cryptocurrency analysis
Created on Dec 17  2017
@author: Dean
"""

#%% 
# IMPORTS & SETUP
# Set the seed for repeatable results (careful with this use)
#print('FIXING SEED FOR REPEATABLE RESULTS')
#from numpy.random import seed
#seed(5)
#from tensorflow import set_random_seed
#set_random_seed(3)

# note that matplotlib notebook isn't working for me
%matplotlib widget


import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))
print(f'Working directory is "{os.getcwd()}"')

import numpy as np
import keras
import FeatureExtraction as FE
import NeuralNet
from Config_CC import GetConfig

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
import pandas as pd

from keras import regularizers

from DataTypes import FeedLoc


def PrintOutDataRanges(r, outData):
    print("\r\noutData 90th Percentile:")
    print(np.percentile(outData, 90, axis=1))
    print("\r\noutData 10th Percentile:")
    print(np.percentile(outData, 10, axis=1))


def PrintInDataRanges(dfs):
    # Print a table. Each column is a feature, each row is a sample
    quantiles = [0.90, 0.10]
    np.zeros((len(dfs), len(dfs[0].columns)))
    values = []

    for i, q in enumerate(quantiles):
        series = []
        for df in dfs:
            ser = df.quantile(q=q)
            ser.name = df.name
            series.append(ser)
        dfq = pd.DataFrame(series)
        dfq.name = q
        values.append(dfq)
        print(f'Input data {q:.2f} quantile')
        print(dfq)



KB.clear_session()

r = ModelResult()  
r.config = GetConfig() 

#At this point, the stock data should have all gaps filled in
#Shape should be (Stocks, Timesteps, Features
#Columns should be ['close', 'high', 'low', 'volumeto']

#r.coinList = ['ETH','BTC','BCH','XRP','LTC','XLM','NEO','EOS','XEM', 'IOT','DOGE','ADA','POT','VET','XLM','ETC']
#r.coinList = ['ETH','BTC','BCH','XRP','LTC']
r.coinList = ['BTC', 'ETH']
#r.coinList = ['ETH']

print('DONE')

# ******************************************************************************
# %% 
# GET DATA

if 0:
    numHours = 24*180
    dfs = cgd.GetHourlyDf(r.coinList, numHours) # a list of data frames
    # To save a data set:
    dateStr = datetime.now().strftime('%Y-%m-%d')
    filehandler = open(f'./indata/dfs_{len(dfs)}coins_{numHours}hours_{dateStr}.pickle', 'wb')
    package = {'dfs':dfs, 'coinList':r.coinList, 'numHours':numHours, 'dateStr':dateStr}
    pickle.dump(package, filehandler)
    filehandler.close()
else:
    # !@#$
#    filehandler = open('dfs_5coins_40days_2018-02-17.pickle', 'rb')
    filehandler = open('./indata/dfs_2coins_4320hours_2021-10-19.pickle', 'rb')
    package = pickle.load(filehandler)
    dfs = package['dfs']
    r.coinList = package['coinList']
    numHours = package['numHours']
    filehandler.close()

for i, df in enumerate(dfs):
    df.name = r.coinList[i]
print('Got data\nDONE')

# ******************************************************************************
# %% 
# PREP DATA

r.sampleCount = len(dfs)
r.timesteps = dfs[0].shape[-2]

prices = np.zeros((r.sampleCount, r.timesteps))
for i in np.arange(r.sampleCount):
    prices[i, :] =  np.array(dfs[i]['close'])

FE.AddLogDiff(r, dfs)
FE.AddVix(r, dfs, prices)
FE.AddRsi(r, dfs)
FE.AddEma(r, dfs)
FE.ScaleLoadedData(dfs) # High, Low, etc

r.inFeatureList = list(dfs[0].columns)
r.inFeatureCount = dfs[0].shape[-1]

# Plot a small sample of the input data
FE.PlotInData(r, dfs, 0, [0, 50])

# Based on the config and the list of features, determine the feed location for each feature
featureList = dfs[0].columns

# INPUT DATA
# inData has 3 separate arrays for 3 separate feed locations
inData = [[] for i in range(FeedLoc.LEN)]
feedLocFeatures = [[] for i in range(FeedLoc.LEN)]

# Determine which features go into which feed locations
for loc in range(FeedLoc.LEN):
    # Find the features that are in this feed location
    feedLocFeatures[loc] = np.zeros_like(featureList, dtype=np.bool)
    for fidx, feature in enumerate(featureList):
        for featureMatch in r.config['feedLoc'][loc]:
            if featureMatch in feature:
                feedLocFeatures[loc][fidx] = True
                break

# Make the input data
for loc in range(FeedLoc.LEN):
    # Make the input data 3D array for this feed location
    inData[loc] = np.zeros((r.sampleCount, r.timesteps, np.sum(feedLocFeatures[loc])))
    for s, df in enumerate(dfs):
        inData[loc][s] = np.array(df.iloc[:,feedLocFeatures[loc]])

r.feedLocFeatures = feedLocFeatures

# Print feed locations for all input data
print("The input feed locations for the features are:")
for loc in range(FeedLoc.LEN):
    print(f"Feed location '{FeedLoc.NAMES[loc]}': {list(featureList[feedLocFeatures[loc]])}")

# OUTPUT DATA
outData = FE.CalcFavScores(r.config, prices)
r.outFeatureCount = outData.shape[-1]

#Scale output values to a reasonable range
#17/12/2017: dividing by 90th percentile was found to be a good scale for SGD
for i in np.arange(r.outFeatureCount):
    outData[:,:,i] /= np.percentile(np.abs(outData[:,:,i]), 90)
    
# Print out data
PrintInDataRanges(dfs)

PrintOutDataRanges(r, outData)
FE.PlotOutData(r, prices, outData, 0)

print(f'Input data (samples={r.sampleCount}, timeSteps={r.timesteps})')

print(f'Output data shape = {outData.shape}')

print('DONE')

# ******************************************************************************
# %% 
# TRAIN

# To reload the NeuralNet function for debugging:
if 1:
    print('Reloading NeuralNet')
    import importlib
    importlib.reload(NeuralNet)


prunedNetwork = True # Pruned: generate multiple candidated and use the best

single = True
if single:
# *****************************************************************************
# Single Run
    r.isBatch = False
    r.batchRunName = ''
    
    # !@#$
    #r.config['neurons'] = [32]
    # r.config['epochs'] = 8
    
    # Scale the input and output data
    thisInData = [arr * r.config['inScale'] for arr in inData]
    
    thisOutData = outData * r.config['outScale']
    if not prunedNetwork:
        NeuralNet.MakeNetwork(r)
        NeuralNet.PrintNetwork(r)
        NeuralNet.TrainNetwork(r, thisInData, thisOutData)
    else:
        NeuralNet.MakeAndTrainPrunedNetwork(r, thisInData, thisOutData)

    NeuralNet.TestNetwork(r, prices, thisInData, thisOutData)

    
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
#            r.sampleCount = len(dfs)
#            r.timesteps = dfs[0].shape[-2]
#
#            prices = np.zeros((r.sampleCount, r.timesteps))
#            for i in np.arange(r.sampleCount):
#                prices[i, :] =  np.array(dfs[i]['close'])
#            
#            FE.AddLogDiff(r, dfs)
#            FE.AddVix(r, dfs, prices)
#            FE.AddRsi(r, dfs)
#            FE.AddEma(r, dfs)
#            FE.ScaleLoadedData(dfs) # High, Low, etc
#            
#            r.inFeatureList = list(dfs[0].columns)
#            r.inFeatureCount = dfs[0].shape[-1]
#            
#            # Convert to a numpy array
#            inData = np.zeros((r.sampleCount, r.timesteps, r.inFeatureCount))
#            for i, df in enumerate(dfs):
#                inData[i] = np.array(df)
#            	
#            # OUTPUT DATA
#            outData = FE.CalcFavScores(r.config, prices)
#            r.outFeatureCount = outData.shape[-1]
#            
#            #Scale output values to a reasonable range
#            #17/12/2017: dividing by 90th percentile was found to be a good scale for SGD
#            for i in np.arange(r.outFeatureCount):
#                outData[:,:,i] /= np.percentile(np.abs(outData[:,:,i]), 90)
            # *********************************************************************
            
            thisInData = inData * r.config['inScale']
            thisOutData = outData * r.config['outScale']
            NeuralNet.MakeAndTrainNetwork(r, thisInData, thisOutData)
            NeuralNet.MakeAndTrainPrunedNetwork(r, thisInData, thisOutData)
            NeuralNet.TestNetwork(r, prices, thisInData, thisOutData)
    
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
    #print('Going to sleep...')
    #os.startfile ('C:\\Users\\Dean\\Desktop\\Sleep.lnk')

print('DONE')