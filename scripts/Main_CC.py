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

from DataTypes import ModelResult, printmd
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

from DataTypes import FeedLoc, printmd


tf.keras.backend.clear_session()

# to force CPU compute:
if 0:
    printmd("**USING ONLY CPU**")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load the input data file
#At this point, the stock data should have all gaps filled in
if not 'dataLoader' in locals():
    inDataFileName = './indata/2021-09-30_price_data_60m.pickle'
    dataLoader = cgd.DataLoader(inDataFileName)
    print('Loaded input file')



# ******************************************************************************
# GET & PREP DATA

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
    FE.AddDivergence(r, dfs)
    FE.ScaleVolume(dfs)

    #FE.AddSpread(r, dfs)
    #FE.PrepHighLowData(dfs) # High, Low, etc

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


def PlotInOutData(r, dfs, inData, outData, prices):
    # Plot a small sample of the input data
    FE.PlotInData(r, dfs, 0, [5000, 10000])

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
printmd('### Imports & data setup DONE')


# ****************************************************************************************************************
# ****************************************************************************************************************
# ****************************************************************************************************************
# %% 
# TRAIN SINGLE
#          d8b                   888          
#          Y8P                   888          
#                                888          
# .d8888b  888 88888b.   .d88b.  888  .d88b.  
# 88K      888 888 "88b d88P"88b 888 d8P  Y8b 
# "Y8888b. 888 888  888 888  888 888 88888888 
#      X88 888 888  888 Y88b 888 888 Y8b.     
#  88888P' 888 888  888  "Y88888 888  "Y8888  
#                            888              
#                       Y8b d88P              
#                        "Y88P"               

# Text font: colossal
# https://patorjk.com/software/taag/#p=display&f=Colossal&t=keras%20tuner


# To reload the NeuralNet function for debugging:
if 1:
    print('Reloading NeuralNet')
    import importlib
    importlib.reload(NeuralNet)


r = ModelResult()
r.config = GetConfig()

r.coinList = ['BTC']
r.numHours = 24*365*5

dfs = dataLoader.GetHourlyDf(r.coinList, r.numHours) # a list of data frames
dfs, inData, outData, prices = PrepData(r, dfs)

#PlotInOutData(r, dfs, inData, outData, prices)

r.isBatch = False
r.batchRunName = ''

r.config['epochs'] = 8

prunedNetwork = False # Pruned: generate multiple candidates and use the best
if not prunedNetwork:
    NeuralNet.MakeNetwork(r)
    NeuralNet.PrintNetwork(r)
    NeuralNet.TrainNetwork(r, inData, outData)
else:
    NeuralNet.MakeAndTrainPrunedNetwork(r, inData, outData)

NeuralNet.TestNetwork(r, prices, inData, outData)

printmd('### Make & train DONE')





# ****************************************************************************************************************
# ****************************************************************************************************************
# ****************************************************************************************************************
#%%
# TRAIN BATCH

# 888               888            888      
# 888               888            888      
# 888               888            888      
# 88888b.   8888b.  888888 .d8888b 88888b.  
# 888 "88b     "88b 888   d88P"    888 "88b 
# 888  888 .d888888 888   888      888  888 
# 888 d88P 888  888 Y88b. Y88b.    888  888 
# 88888P"  "Y888888  "Y888 "Y8888P 888  888 

#
r = ModelResult()
r.config = GetConfig() 
r.coinList = ['BTC']
r.numHours = 24*365*3

r.config['epochs'] = 64

# Batch changes
# Val1: rows. Val2:" columns"
bat1Name = 'Trial'
bat1Val = [1,2,3,4,5]

bat2Name = 'Dropout'
bat2Val = [0., 0.1, 0.2, 0.35]


# Boilerplate...
bat1Len = len(bat1Val)
bat2Len = len(bat2Val)

results = [0]*bat2Len
r.isBatch = True
r.batchName = datetime.now().strftime('%Y-%m-%d_%h_') + '_' + bat1Name + '_' + bat2Name
startR = r

printmd('# Batch run START')
trialCount = 0

for idx2, val2 in enumerate(bat2Val):
    results[idx2] = [0]*bat1Len
    
    for idx1, val1 in enumerate(bat1Val):
        tf.keras.backend.clear_session() 
        results[idx2][idx1] = copy.deepcopy(startR)
        r = results[idx2][idx1]
        
        printmd(f'### BATCH RUN ({idx2}, {idx1}). Trial {trialCount} / {bat1Len * bat2Len}')
        r.batchRunName = '{bat2Name}:{val2}, {bat1Name}:{val1}'.format(bat2Name, val2, bat1Name, val1)
        print(r.batchRunName)
            
        # *****************************
        # Change for this batch
        r.config['dropout'] = val2
        # *****************************
        
        dfs = dataLoader.GetHourlyDf(r.coinList, r.numHours) # a list of data frames
        dfs, inData, outData, prices = PrepData(r, dfs)
        
        NeuralNet.MakeNetwork(r)
        NeuralNet.TrainNetwork(r, inData, outData, plotMetrics=False)

        #NeuralNet.MakeAndTrainPrunedNetwork(r, inData, outData, drawPlots=False, candidates = 3, trialEpochs = 16)
        NeuralNet.TestNetwork(r, prices, inData, outData, drawPlots=False)

        trialCount += 1

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
#%%
# BATCH: PLOT GRID

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numbers
from NeuralNet import PlotTrainMetrics

# Uncomment either option A or B
# OPTION A: batVal1 along x (columns), batVal2 along y (rows)
#plt.figure(figsize=(bat1Len*5,bat2Len*3)); p = 1
#for idx2 in range(bat2Len):
#    for idx1 in range(bat1Len):
#        plt.subplot(bat2Len, bat1Len, p)
        
# OPTION B: batVal2 along x (columns), batVal1 along y (rows)
fig, axs = plt.subplots(bat1Len, bat2Len, figsize=(bat2Len*5,bat1Len*3)); p = 1
fig.tight_layout()
minY = 9e9
maxY = -9e9
for idx1 in range(bat1Len):
    rowAx = []
    for idx2 in range(bat2Len):
        ax = axs[idx1, idx2]
        r = results[idx2][idx1] # Pointer for brevity
        
        (thisMaxY, thisMinY) = PlotTrainMetrics(r, ax)
        maxY = max(maxY, thisMaxY)
        minY = min(minY, thisMinY)
        
        ax.set_title(f'{bat2Name}:{bat2Val[idx2]}, {bat1Name}:{bat1Val[idx1]}', fontdict={'fontsize':10})
        #ax.set_yscale('log')
        ax.grid()
        
        print('{}:{}, {}:{}'.format(bat2Name, bat2Val[idx2], bat1Name, bat1Val[idx1]))
        print('Train Score: {:5}\nTest Score: {:5} (1=neutral)'.format(r.trainScore, r.testScore))

maxY = round(maxY+0.05, 1)
minY = round(minY-0.05, 1)
# Set all to have the same axes
for idx1 in range(bat1Len):
    for idx2 in range(bat2Len):
        axs[idx1,idx2].set_ylim(bottom=minY, top=maxY)
        axs[idx1,idx2].set_xlim(left=0, right=r.config['epochs']-1)
plt.show()



# *****************************************************************************
#%%
# LINE PLOTS

# 1 PLOT, MULTIPLE LINES
def DrawPlot(valA, valB, nameA, nameB, data, nameY):
    # valA is the x axis
    if (not isinstance(valA[0], numbers.Number) or len(valA) < 3):
        return
    fig, ax = plt.subplots(figsize=(7,4))
    fig.tight_layout()
    ax.plot(valA, data)
    diffA = np.diff(valA)
    if diffA[-1]/diffA[0] > 5:
        ax.set_xscale('log')
        ax.set_xticks(valA)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(nameA)
    ax.set_ylabel(nameY)
    ax.set_title('{} vs {} (Legend = {})'.format(nameY, nameA, nameB))
    ax.legend(valB)
    plt.show()

# Test Score vs bat1Val
data = np.array([[r.testScore for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Test Score')

# Test Score vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Test Score')

# Train Score vs bat1Val
data = np.array([[r.trainScore for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Train Score')

# Train Score vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Train Score')

# Training Time vs bat1Val
data = np.array([[r.trainTime for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Training Time')

# Training Time vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Training Time')

## PLOT ALL PREDICTIONS
#for idx1 in range(bat1Len):
#    for idx2 in range(bat2Len):
#        r = results[idx2][idx1] # Pointer for brevity
#        TestNetwork(r, prices, thisInData, thisOutData, tInd)



# ****************************************************************************************************************
# ****************************************************************************************************************
# ****************************************************************************************************************
# %%
# KERAS TUNER

# 888                                             888                                       
# 888                                             888                                       
# 888                                             888                                       
# 888  888  .d88b.  888d888 8888b.  .d8888b       888888 888  888 88888b.   .d88b.  888d888 
# 888 .88P d8P  Y8b 888P"      "88b 88K           888    888  888 888 "88b d8P  Y8b 888P"   
# 888888K  88888888 888    .d888888 "Y8888b.      888    888  888 888  888 88888888 888     
# 888 "88b Y8b.     888    888  888      X88      Y88b.  Y88b 888 888  888 Y8b.     888     
# 888  888  "Y8888  888    "Y888888  88888P'       "Y888  "Y88888 888  888  "Y8888  888     

printmd("## Keras tuner")
import keras_tuner as kt

r = ModelResult()
r.config = GetConfig() 
r.coinList = ['BTC', 'ETH']
r.numHours = 24*365*3
r.config['epochs'] = 64

dfs = dataLoader.GetHourlyDf(r.coinList, r.numHours) # a list of data frames
dfs, inData, outData, prices = PrepData(r, dfs)


def build_model(hp):
    outRangeStart = hp.Int('outRangeStart', min_value=1, max_value=144, sampling='log')
    r.config['outputRanges'] = [[outRangeStart, outRangeStart*2]]

    # Changing model
    #r.config['convKernelSz'] = hp.Int("convKernelSz", min_value=3, max_value=256, sampling='log')

    # lstmLayerCount = hp.Int("lstmLayerCount", min_value=1, max_value=3)
    # r.config['lstmWidths'] = []
    # for i in range(lstmLayerCount):
    #     r.config['lstmWidths'].append(hp.Int(f"lstm_{i}", min_value=8, max_value=512, sampling='log'))
    
    # r.config['bottleneckWidth'] = hp.Int(f"bottleneckWidth", min_value=8, max_value=512, sampling='log')

    
    NeuralNet.MakeNetwork(r)
    return r.model

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective("val_score_sq_any", direction="max"),
    max_trials=10,
    executions_per_trial=1, # number of attempts with the same settings
    overwrite=True,
    directory="keras_tuner",
    project_name="fortune_test",
)

tuner.search_space_summary()

fitArgs, checkpointCb, printoutCb = NeuralNet.PrepTrainNetwork(r, inData, outData)

fitArgs['verbose'] = 1

# Start
start = time.time()
tuner.search(**fitArgs)
end = time.time()
r.trainTime = end-start
print(f'Tuning Time (h:m:s)= {NeuralNet.SecToHMS(r.trainTime)}.  {r.trainTime:.1f}s')

#tuner.results_summary(). # This is very poorly formatted

#%%
# Keras tuner results into pandas
trials = tuner.oracle.get_best_trials(9999)
dfData = [copy.deepcopy(t.hyperparameters.values) for t in trials]
for i, row in enumerate(dfData):
    row['rank'] = i
    row['score'] = trials[i].score
    row['id'] = trials[i].trial_id

df = pd.DataFrame(dfData)
df.set_index('id')
print(df)

# Plot tuner results
hpNames = trials[0].hyperparameters.values.keys()
fig, axs = plt.subplots(len(hpNames), 1, figsize=(r.config['plotWidth'] , 4 * len(hpNames)))
if len(hpNames) == 1:
    axs = [axs]

fig.tight_layout()
for ax, hpName in zip(axs, hpNames):
    ax.plot(df[hpName], df['score'], 'x', label=hpName)
    #ax.set_title(hpName)
    ax.legend(labels=[hpName], loc='lower right')
    ax.grid()
plt.show()

