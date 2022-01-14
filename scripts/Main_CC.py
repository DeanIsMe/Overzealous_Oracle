# -*- coding: utf-8 -*-
"""
Main - Cryptocurrency analysis
Created on Dec 17  2017
@author: Dean
"""

#%% 
# IMPORTS & SETUP

# note that "matplotlib notebook" isn't working for me
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

from DataTypes import ModelResult, printmd, SecToHMS
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
    # feedLocFeatures is a list of 3 boolean arrays. 
    # Has a bool entry for every column in dfs
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

# Set the seed for repeatable results (careful with this use)
# print('FIXING SEED FOR REPEATABLE RESULTS')
# from numpy.random import seed
# seed(5)
# tf.random.set_seed(5)

printmd("## Start single train")
r = ModelResult()
r.config = GetConfig()


r.config['epochs'] = 32
r.config['revertToBest'] = False


dfs = dataLoader.GetHourlyDf(r.config['coinList'], r.config['numHours']) # a list of data frames
dfs, inData, outData, prices = PrepData(r, dfs)

#PlotInOutData(r, dfs, inData, outData, prices)

r.isBatch = False
r.batchRunName = ''

prunedNetwork = False # Pruned: generate multiple candidates and use the best
if not prunedNetwork:
    NeuralNet.MakeNetwork(r)
    #NeuralNet.PrintNetwork(r)
    NeuralNet.TrainNetwork(r, inData, outData)
else:
    NeuralNet.MakeAndTrainPrunedNetwork(r, inData, outData)

NeuralNet.TestNetwork(r, prices, inData, outData)

printmd('### Make & train DONE')





# ** **************************************************************************************************************
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

r.config['epochs'] = 64

# Batch changes
# Val1: rows. Val2: columns
bat1Name = 'BatchNorm'
bat1Val = [False, True]

bat2Name = 'Trial'
bat2Val = [0, 1, 2]

# Boilerplate...
bat1Len = len(bat1Val)
bat2Len = len(bat2Val)

results = [0]*bat2Len

r.isBatch = True
r.batchName = datetime.now().strftime('%Y-%m-%d_%H%M') + '_' + bat1Name + '_' + bat2Name
batchDir = f"batches/{r.batchName}/"
os.makedirs(os.path.dirname(batchDir), exist_ok=True)
startR = r

printmd('# Batch START')
printmd(f"## {r.batchName}")
print(r.batchName)
print(f"bat1Name = {bat1Name}")
print(f"bat1Val  = {bat1Val}")
print(f"bat2Name = {bat2Name}")
print(f"bat2Val  = {bat2Val}")

trialCount = 0
totalTrials = bat1Len * bat2Len
batchStartTime = time.time()

for idx2, val2 in enumerate(bat2Val):
    results[idx2] = [0]*bat1Len
    
    for idx1, val1 in enumerate(bat1Val):
        tf.keras.backend.clear_session() 
        results[idx2][idx1] = copy.deepcopy(startR)
        r = results[idx2][idx1]
        
        r.batchRunName = f'{bat2Name}:{val2}, {bat1Name}:{val1}'.format(bat2Name, val2, bat1Name, val1)
        printmd(f'### Batch Run {trialCount} / {totalTrials} ({idx2}, {idx1})')
        printmd(f"**{r.batchRunName}**")

        if trialCount > 0:
            elapsed = time.time() - batchStartTime
            remaining = elapsed / (trialCount) * (totalTrials - trialCount)
            print(f"{SecToHMS(elapsed)} elapsed.  ~{SecToHMS(remaining)} remaining.")

        # *****************************
        # Change for this batch

        r.config['batchNorm'] = val1
        # *****************************
        
        dfs = dataLoader.GetHourlyDf(r.config['coinList'], r.config['numHours'], verbose=0) # a list of data frames
        dfs, inData, outData, prices = PrepData(r, dfs)
        
        NeuralNet.MakeNetwork(r)
        NeuralNet.TrainNetwork(r, inData, outData, plotMetrics=False)

        #NeuralNet.MakeAndTrainPrunedNetwork(r, inData, outData, drawPlots=False, candidates = 3, trialEpochs = 16)
        NeuralNet.TestNetwork(r, prices, inData, outData, drawPlots=False)

        trialCount += 1

print(f"\n\nBATCH RUN FINISHED!\n Duration: {SecToHMS(time.time() - batchStartTime)}")

# SAVE THE DATA
# Clear the model so that 'r' can pickle
models = [0] * bat2Len
for idx2, rList in enumerate(results):
    models[idx2] = [0]*bat1Len
    for idx1, r in enumerate(rList):
        r = results[idx2][idx1]
        models[idx2][idx1] = r.model
        r.model = None


filehandler = open(batchDir + f"{r.batchName}.pickle", 'wb') 
pickle.dump(results, filehandler)
filehandler.close()

# Copy the model back in
for idx2, rList in enumerate(results):
    for idx1, r in enumerate(rList):
        results[idx2][idx1].model = models[idx2][idx1]


filehandler = open(batchDir + f"config.txt", 'w') 
filehandler.writelines(([f"{k:>20s} : {r.config[k]},\n" for k in r.config.keys()]))
filehandler.close()

filehandler = open(batchDir + f"r.txt", 'w') 
filehandler.write(str(vars(r)))
filehandler.close()

#Go to sleep
#print('Going to sleep...')
#os.startfile ('C:\\Users\\Dean\\Desktop\\Sleep.lnk')

printmd('## Batch run DONE')



# # *****************************************************************************

# BATCH: PLOT GRID

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numbers
from NeuralNet import PlotTrainMetrics

# batVal2 along x (columns), batVal1 along y (rows)

# if the 2nd variable is named 'trial', then stack lines on top
columns = 1 if bat2Name.lower() == 'trial' else bat2Len
    

fig, axs = plt.subplots(bat1Len, columns, figsize=(columns*5,bat1Len*3)); p = 1
fig.tight_layout()
def getAx(idx1, idx2):
    if bat1Len == 1:
        if columns == bat2Len: return axs[idx2]
        else: return axs
    if columns == bat2Len: return axs[idx1, idx2]
    else: return axs[idx1]

minY = 9e9
maxY = -9e9
for idx1 in range(bat1Len):
    rowAx = []
    for idx2 in range(bat2Len):
        ax = getAx(idx1, idx2)
        r = results[idx2][idx1] # Pointer for brevity
        
        (thisMaxY, thisMinY) = PlotTrainMetrics(r.trainHistory, ax, legend=((idx1+idx2)==0))
        maxY = max(maxY, thisMaxY)
        minY = min(minY, thisMinY)
        
        ax.set_title(f'{bat2Name}:{bat2Val[idx2]}, {bat1Name}:{bat1Val[idx1]}', fontdict={'fontsize':10})
        #ax.set_yscale('log')
        

maxY = round(maxY+0.05, 1)
minY = round(minY-0.05, 1)
# Set all to have the same axes limits
for idx1 in range(bat1Len):
    for idx2 in range(columns):
        ax = getAx(idx1, idx2)
        ax.set_ylim(bottom=minY, top=maxY)
        ax.set_xlim(left=0, right=r.config['epochs']-1)
plt.show()
plt.savefig(batchDir + "plot_trainMetrics.png")



# *****************************************************************************
# LINE PLOTS

#%%
def DrawPlotArgs(valA, valB, nameA, nameB, data, nameY):
    """
    Prepares and returns a dictionary with arguments for DrawPlot
    Returns None, or a dictionary with arguments for DrawPlot
    """
    if (not isinstance(valA[0], numbers.Number) or len(valA) < 3):
        return None
    return {'valA':valA,
    'valB':valB,
    'nameA':nameA,
    'nameB':nameB,
    'data':data,
    'nameY':nameY}

def DrawPlot(ax, valA, valB, nameA, nameB, data, nameY):
    # Plots onto an existing axis
    # valA is the x axis
    ax.plot(valA, data)
    diffA = np.diff(valA)
    if diffA[-1]/diffA[0] > 5:
        ax.set_xscale('log')
        ax.set_xticks(valA)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    # Add linear trendline
    z = np.polyfit(valA, np.mean(data, axis=1) ,1)
    p = np.poly1d(z)
    ax.plot(valA, p(valA),ls=':', c='grey')
    # the line equation:
    print(f"Score vs {nameA:15s}. y= {z[0]:9.6f}x + {z[1]:9.6f}")

    #ax.set_xlabel(nameA)
    ax.set_ylabel(nameY)
    ax.set_title('{} vs {} (Legend = {})'.format(nameY, nameA, nameB), fontdict={'fontsize':10})
    ax.legend(valB)
    ax.grid(True)
    return ax

plots = []
# Test Score vs bat1Val
data = np.array([[np.max(r.trainHistory['val_score_sq_any']) for r in results[idx2]] for idx2 in range(bat2Len)])
plots.append(DrawPlotArgs(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'TestScoreAny'))

# Test Score vs bat2Val
plots.append(DrawPlotArgs(bat2Val, bat1Val, bat2Name, bat1Name, data, 'TestScoreAny'))

# Train Score vs bat1Val
data = np.array([[np.max(r.trainHistory['score_sq_any']) for r in results[idx2]] for idx2 in range(bat2Len)])
plots.append(DrawPlotArgs(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'TrainScoreAny'))

# Train Score vs bat2Val
plots.append(DrawPlotArgs(bat2Val, bat1Val, bat2Name, bat1Name, data, 'TrainScoreAny'))

# Training Time vs bat1Val
data = np.array([[r.trainTime / len(r.trainHistory['loss']) for r in results[idx2]] for idx2 in range(bat2Len)])
plots.append(DrawPlotArgs(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'SecPerEpoch'))

# Training Time vs bat2Val
plots.append(DrawPlotArgs(bat2Val, bat1Val, bat2Name, bat1Name, data, 'SecPerEpoch'))

# Remove 'None' values
plots = [p for p in plots if p is not None]

#Now that I know how many plots there are, plot it!
fig, axs = plt.subplots(len(plots), 1, figsize=(5,3*len(plots)))
fig.tight_layout()

for i, args in enumerate(plots):
    DrawPlot(axs[i], **args)

plt.show()
plt.savefig(batchDir + "plot_correlation.png")

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
r.config['epochs'] = 8


class HistData:
    """Tiny class (struct) to store history data during keras training
    """
    def __init__(self):
        self.latestTrialId = None
        self.allHist = []


class MyHyperModel(kt.HyperModel):
    """[summary]
    Default class is in:
    venv\Lib\site-packages\keras_tuner\engine\hypermodel.py

    """
    

    # def __init__(self, *args, **kwargs):
    #     super(MyHyperModel, self).__init__(*args, **kwargs)
    #     self.allHist = []

    def setHistData(self, histData:HistData):
        self.histData = histData

    def build(self, hp):
        # output range
        # outRangeStart = hp.Int('outRangeStart', min_value=1, max_value=144, sampling='log')
        # r.config['outputRanges'] = [[outRangeStart, outRangeStart*2]]

        # Output scaling
        # outliers = hp.Float("outliers", -0.3, 3.)
        # if outliers < 0.:
        #     r.config['binarise'] = 0
        #     r.config['ternarise'] = 0
        # elif outliers < 1.:
        #     r.config['binarise'] = outliers
        #     r.config['ternarise'] = 0
        # else:
        #     r.config['binarise'] = 0
        #     r.config['ternarise'] = (outliers - 1.) * 2.5
        #     r.config['selectivity'] = hp.Float("selectivity", 1., 3.)

        # Changing model
        #r.config['convKernelSz'] = hp.Int("convKernelSz", min_value=3, max_value=256, sampling='log')

        maxStepsPast = 100 * 24

        r.config['vixNumPastRanges'] = hp.Int("vixFeatures", min_value=0, max_value=2) # number of ranges to use
        r.config['vixMaxPeriodPast'] = maxStepsPast
        
        # RSI - Relative Strength Index
        #rsiFeatures = hp.Int("rsiFeatures", min_value=0, max_value=2)
        #r.config['rsiWindowLens'] = list(np.geomspace(start=5, stop=maxStepsPast, num=rsiFeatures, dtype=int)) # The span of the EMA calc for RSI. E.g. 24,96 for 2 RSI features with 24 and 96 steps
        r.config['rsiWindowLens'] = []
        
        # # Exponential Moving Average
        # emaFeatures = hp.Int("emaFeatures", min_value=0, max_value=5, sampling='log')
        # r.config['emaLengths'] = list(np.geomspace(start=5, stop=180*24, num=emaFeatures, dtype=int))

        # I define divergence as the price relative to the moving average of X points
        dvgFeatures = hp.Int("dvgFeatures", min_value=0, max_value=5)
        r.config['dvgLengths'] = np.geomspace(start=5, stop=maxStepsPast, num=dvgFeatures, dtype=int)


        dfs = dataLoader.GetHourlyDf(r.config['coinList'], r.config['numHours'], verbose=0) # a list of data frames
        self.dfs, self.inData, self.outData, self.prices = PrepData(r, dfs)
        NeuralNet.MakeNetwork(r)
        return r.model

    def fit(self, hp, model, *args, **kwargs):
        """Train the model.

        Args:
            hp: HyperParameters.
            model: `keras.Model` built in the `build()` function.
            **kwargs: All arguments passed to `Tuner.search()` are in the
                `kwargs` here. It always contains a `callbacks` argument, which
                is a list of default Keras callback functions for model
                checkpointing, tensorboard configuration, and other tuning
                utilities. If `callbacks` is passed by the user from
                `Tuner.search()`, these default callbacks will be appended to
                the user provided list.

        Returns:
            A `History` object, which is the return value of `model.fit()`, a
            dictionary, or a float.

            If return a dictionary, it should be a dictionary of the metrics to
            track. The keys are the metric names, which contains the
            `objective` name. The values should be the metric values.

            If return a float, it should be the `objective` value.
        """
        fitArgs, checkpointCb, printoutCb = NeuralNet.PrepTrainNetwork(r, self.inData, self.outData)
        # keras tuner overrides the callbacks passed to 'fit()'. Combine any kwargs with
        # the args generated from my PrepTrainNetwork.
        for key in kwargs.keys():
            if key == 'callbacks':
                fitArgs['callbacks'] = fitArgs['callbacks'] + kwargs['callbacks']
            else:
                fitArgs[key] = kwargs[key]
        fitArgs['verbose'] = 0
        hist = model.fit(*args, **fitArgs)
        histData.allHist.append({'id':histData.latestTrialId, 'hp':hp.values, 'history':hist.history})
        return hist


class MyRandomTuner(kt.RandomSearch):
    """I created this custom class solely to get the trial ID
    """
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def setHistData(self, histData:HistData):
        self.histData = histData

    def on_trial_begin(self, trial):
        """Called at the beginning of a trial.
        """
        histData.latestTrialId = trial.trial_id
        super().on_trial_begin(trial)


log_dir = "logs/" + datetime.now().strftime('%Y-%m-%d_%H%M') + '/'
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    embeddings_freq=1,
    write_graph=True,
    update_freq='batch')


histData = HistData()
hyperModel = MyHyperModel()
hyperModel.setHistData(histData)

tuner = MyRandomTuner(
    hypermodel=hyperModel,
    objective=kt.Objective("val_fitness", direction="max"),
    max_trials=10,
    executions_per_trial=1, # number of attempts with the same settings
    overwrite=True,
    directory="keras_tuner",
    project_name="fortune_test",
)
tuner.setHistData(histData)

tuner.search_space_summary()

# SEARCH
start = time.time()
tuner.search(callbacks=[tensorboard_cb])
end = time.time()
r.trainTime = end-start
print(f'Tuning Time (h:m:s)= {SecToHMS(r.trainTime)}.  {r.trainTime:.1f}s')
printmd("## keras tuner done")

#tuner.results_summary(). # This is very poorly formatted
#%%
# PRINT RESULTS
# Keras tuner results into pandas
dfData = []
metrics_max = ['val_fitness', 'fitness', 'val_score_sq_any', 'score_sq_any']
score_metric = 'val_fitness' # The metric name that is the objective. Assumed maximized
for idx, trial in enumerate(histData.allHist):
    d = {}
    d['idx'] = idx
    d['trial_id'] = trial['id']
    d.update(trial['hp']) # add hyperparameters
    # Add metrics
    d['score'] = np.max(trial['history'][score_metric])
    d['best_epoch'] = np.argmax(trial['history'][score_metric])
    d['val_penalty_at_best'] = trial['history']['val_penalty'][d['best_epoch']]
    for metric in metrics_max:
        d['max_' + metric] = np.max(trial['history'][metric])
    dfData.append(d)

df = pd.DataFrame(dfData)
df.set_index('idx')
print(df)

#%%
# Plot train metrics of the best performing
best_trial_idx = df.loc[df['score'].idxmax(),'idx']
NeuralNet.PlotTrainMetrics(histData.allHist[best_trial_idx]['history'])



# Plot tuner results
hpNames = histData.allHist[0]['hp'].keys()
fig, axs = plt.subplots(len(hpNames), 1, figsize=(r.config['plotWidth'] , 4 * len(hpNames)))
if len(hpNames) == 1:
    axs = [axs]

fig.tight_layout()
for ax, hpName in zip(axs, hpNames):
    ax.plot(df[hpName], df['score'], 'x', label=hpName)
    #ax.set_title(hpName)
    ax.set_title(hpName, fontdict={'fontsize':10})
    ax.grid()
plt.show()

# TODO add trendlines