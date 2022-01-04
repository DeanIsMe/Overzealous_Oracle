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

import tensorflow
os.chdir(os.path.dirname(os.path.dirname(__file__)))
print(f'Working directory is "{os.getcwd()}"')

import numpy as np
import tensorflow as tf
from tensorflow import keras

import FeatureExtraction as FE
import NeuralNet
from Config_CC import GetConfig

from DataTypes import ModelResult
from TestSequences import GetInSeq
import InputData as indata
import copy

import datetime

import matplotlib.pyplot as plt
import time
import Crypto_GetData as cgd
import pickle
from datetime import datetime
import pandas as pd

from DataTypes import FeedLoc

from IPython.display import Markdown, display
def printmd(string, color=None):
    if color is None:
        display(Markdown(string))
    else:
        colorstr = "<span style='color:{}'>{}</span>".format(color, string)
        display(Markdown(colorstr))


tf.keras.backend.clear_session()

# to force CPU compute:
if 0:
    printmd("**USING ONLY CPU**")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

r = ModelResult()
r.config = GetConfig() 


# Load the input data file
#At this point, the stock data should have all gaps filled in
inDataFileName = './indata/2021-09-30_price_data_60m.pickle'
dataLoader = cgd.DataLoader(inDataFileName)
print('Loaded input file')


printmd('### Imports DONE')

# ******************************************************************************
# %% 
# GET & PREP DATA

#r.coinList = ['ETH','BTC','BCH','XRP','LTC','XLM','NEO','EOS','XEM', 'IOT','DOGE','ADA','POT','VET','XLM','ETC']
#r.coinList = ['ETH','BTC','BCH','XRP','LTC']
r.coinList = ['BTC', 'ETH']
#r.coinList = ['ETH']
r.numHours = 24*365*5


def PrepData(r:ModelResult, dfs:list):
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

    # Based on the config and the list of features, determine the feed location for each feature
    featureList = dfs[0].columns

    # INPUT DATA
    # inData has 3 separate arrays for 3 separate feed locations
    inData = [[] for i in range(FeedLoc.LEN)]
    feedLocFeatures = [[] for i in range(FeedLoc.LEN)]

    # Determine which features go into which feed locations
    for loc in range(FeedLoc.LEN):
        # Find the features that are in this feed location
        feedLocFeatures[loc] = np.zeros_like(featureList, dtype=bool)
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


    # OUTPUT DATA
    outData = FE.CalcFavScores(r.config, prices)
    r.outFeatureCount = outData.shape[-1]

    #Scale output values to a reasonable range
    #17/12/2017: dividing by 90th percentile was found to be a good scale for SGD
    for i in np.arange(r.outFeatureCount):
        outData[:,:,i] /= np.percentile(np.abs(outData[:,:,i]), 90)

    # Scale the input data
    if r.config['inScale'] != 1.:
        inData = [arr * r.config['inScale'] for arr in inData]

    # Scale the output data
    if r.config['outScale'] != 1.:
        outData = outData * r.config['outScale']

    return dfs, inData, outData, prices


dfs = dataLoader.GetHourlyDf(r.coinList, r.numHours) # a list of data frames
dfs, inData, outData, prices = PrepData(r, dfs)


# Plot a small sample of the input data
FE.PlotInData(r, dfs, 0, [500, 700])

# Print info about in & out data:
print("The input feed locations for the features are:")
for loc in range(FeedLoc.LEN):
    print(f"Feed location '{FeedLoc.NAMES[loc]}': {list(dfs[0].columns[r.feedLocFeatures[loc]])}")

# Print data ranges
FE.PrintInOutDataRanges(dfs, outData)

FE.PlotOutData(r, prices, outData, 0)

print(f'Input data (samples={r.sampleCount}, timeSteps={r.timesteps})')

print(f'Output data shape = {outData.shape}')

# Data shape should be (Stocks, Timesteps, Features)
printmd('### Prep data DONE')

# ******************************************************************************
# %% 
# TRAIN SINGLE

# To reload the NeuralNet function for debugging:
if 1:
    print('Reloading NeuralNet')
    import importlib
    importlib.reload(NeuralNet)

prunedNetwork = False # Pruned: generate multiple candidates and use the best

single = True
if single:
# *****************************************************************************
# Single Run
    r.isBatch = False
    r.batchRunName = ''
    
    # !@#$
    #r.config['lstmWidths'] = [64]
    r.config['epochs'] = 8
    
    if not prunedNetwork:
        NeuralNet.MakeNetwork(r)
        NeuralNet.PrintNetwork(r)
        NeuralNet.TrainNetwork(r, inData, outData)
    else:
        NeuralNet.MakeAndTrainPrunedNetwork(r, inData, outData)

    NeuralNet.TestNetwork(r, prices, inData, outData)
    
    printmd('### Make & train DONE')
else:
    print('Run next cell for batch...')
    

#%%
# TRAIN BATCH
if not single:
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
            tf.keras.backend.clear_session() 
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
            
            dfs = dataLoader.GetHourlyDf(r.coinList, r.numHours) # a list of data frames
            dfs, inData, outData, prices = PrepData(r, dfs)
            
            NeuralNet.MakeAndTrainNetwork(r, inData, outData)
            #NeuralNet.MakeAndTrainPrunedNetwork(r, inData, outData)
            NeuralNet.TestNetwork(r, prices, inData, outData)
    
    print('\n\nBATCH RUN FINISHED!\n')
    # SAVE THE DATA
    # Clear the model so that 'r' can pickle
    models = [0] * bat2Len
    for idx2, rList in enumerate(results):
        models[idx2] = [0]*bat1Len
        for idx1, r in enumerate(rList):
            r = results[idx2][idx1]
            models[idx2][idx1] = r.model
            r.model = 0
    
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

printmd('## Batch run DONE')


# *****************************************************************************
# %%
# KERAS TUNER
printmd("## Keras tuner")
import keras_tuner as kt

r.coinList = ['BTC', 'ETH']
r.numHours = 24*365*3
r.config['epochs'] = 32

dfs = dataLoader.GetHourlyDf(r.coinList, r.numHours) # a list of data frames
dfs, inData, outData, prices = PrepData(r, dfs)


def build_model(hp):
    outRangeStart = hp.Int('outRangeStart', min_value=1, max_value=72, sampling='log')
    r.config['outputRanges'] = [[outRangeStart, outRangeStart*2]]
    # Changing model
    # r.config['convKernelSz'] = hp.Int("convKernelSz", min_value=3, max_value=256, sampling='log')

    # lstmLayerCount = hp.Int("lstmLayerCount", min_value=1, max_value=3)
    # r.config['lstmWidths'] = []
    # for i in range(lstmLayerCount):
    #     r.config['lstmWidths'].append(hp.Int(f"lstm_{i}", min_value=8, max_value=512, sampling='log'))
    
    # r.config['bottleneckWidth'] = hp.Int(f"bottleneckWidth", min_value=8, max_value=512, sampling='log')

    
    NeuralNet.MakeNetwork(r)
    return r.model

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective("val_score_abs", direction="max"),
    max_trials=20,
    executions_per_trial=1, # number of attemps with the same settings
    overwrite=True,
    directory="keras_tuner",
    project_name="fortune_test",
)

tuner.search_space_summary()

fitArgs, checkpointCb, printoutCb = NeuralNet.PrepTrainNetwork(r, inData, outData)

# Start
start = time.time()
tuner.search(**fitArgs)
end = time.time()
r.trainTime = end-start
print(f'Tuning Time (h:m:s)= {NeuralNet.SecToHMS(r.trainTime)}.  {r.trainTime:.1f}s')

tuner.results_summary()

# %%
import tensorflow.keras as tfk
tfk.Model
tfk.Model.fit