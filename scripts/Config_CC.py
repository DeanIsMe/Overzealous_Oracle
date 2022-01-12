# -*- coding: utf-8 -*-
# Cryptocurrency configuration

from DataTypes import FeedLoc
import numpy as np

#==============================================================================
# GetConfig
# Defines and returns the configuration dict
def GetConfig():
    config = dict()
    
    # ****************************
    # INPUT DATA / FEATURES

    # Volatility
    config['vixNumPastRanges'] = 3 # number of ranges to use
    config['vixMaxPeriodPast'] = 24*14
    
    # RSI - Relative Strength Index
    config['rsiWindowLens'] = [24, 96] # The span of the EMA calc for RSI
    
    # Exponential Moving Average
    config['emaLengths'] = [] # The span of the EMAs

    # I define divergence as the price relative to the moving average of X points
    config['dvgLengths'] = np.geomspace(start=10, stop=365, num=5, dtype=int)



    # ****************************
    # OUTPUT (TARGET) DATA

    # config['outputRanges'] = [[1,5], [6,25], [26,125]] # The ranges over which to calculate output scores
    # Starts at 1
    # I'm using hourly data
    config['outputRanges'] = [[1,5], [6,25], [26,125]]
    # Also defines the number of output periods

    # Favourability Score: CompareToFutureData
    # How many steps to exclude from training because the 'to buy' score is not well defined
    config['excludeRecentSteps'] = 50 # Tradeoff between how recent, and accuracy to 'to Buy' score
    
    # When the date is within 'excludeRecentSteps' of the end of the data, then
    # the 'favourability' score cannot be completely calculated.
    # When pullUncertainYTo0 is true, the favourability score will be pulled
    # more towards zero as the more and more dates are missing. When it's
    # false, the score will be calculated assuming that the price does not vary
    # at all from the last day that data exists
    config['pullUncertainYTo0'] = True

    # Binarise
    # Minimise outliers reduces the size of outliers in output favourability score
    # The binarise =0.1 gives almost no reduction
    # =0.8 gives a huge reduction of outliers
    # =3 makes the data almost binary
    # 0.4 is a good tradeoff for minimising outliers
    config['binarise'] = 0
    
    # Ternarise
    # Transform towards a ternary buy/sell/neutral
    # ternarise = 10 (strict ternary). 
    # = 0.5 is less strict, but approaches binary
    # 1-3 seems preferable
    # Binarise and ternarse must not both be non-zero
    config['ternarise'] = 0

    # selectivity only applies when ternarise is non-zero. Higher=more often neutral
    # selectivity 1=frequent buy/sell signals. 3=very picky buy/sell
    config['selectivity'] = 2
    # A good balance seems to be ternarise = 1, selectivity = 2


    # ****************************
    # NEURAL NET (MODEL)

    config['bottleneckWidth'] = 64 # A dense layer is added before the LSTM to reduce the LSTM size
    config['lstmWidths'] = [96] # Number of neurons in each LSTM layer. They're cascaded.
    # The 3 parameters below can be a list (one value per conv layer), or a scalar (apply to all conv layers)
    # The num of conv layers will be the greatest number of valid layers
    # If any parameter is empty ([] or 0), then there will be no convolutional layers
    config['convDilation'] = [1,2,4,8,16,32,64,128] # Time dilation factors. 
    config['convFilters'] = [80,75,70,65,60,50,40,30] # Number of filters per layer. List or scalar
    config['convKernelSz'] = 10 # Kernel size per filter
    config['denseWidths'] = [] # [256, 128, 64, 32, 16] # These layers are added in series after LSTM and before output layers. Default: none
    config['batchNorm'] = True
    config['useGru'] = True # If true, swaps the LSTM with a GRU


    # ****************************
    # TRAINING DATA

    config['dataRatios'] = [0.80, 0.20 ,0.00] # Training, Validation, Testing
    #note: 03/02/2018 I am using the same set for validation and testing
    # I'm just using the 'test' set to generate state for prediction

    config['evaluateBuildStatePoints'] = 500 # The number of timesteps used to build state when predicting values for model validation during training

    # Input data and output data are divided by 90th percentile. Then, they
    # are multiplied by the 'scale'. This has a massive impact on the training
    # and results
    config['inScale'] = 1.
    config['outScale'] = 1.

    # feedLoc indicates the location at which the data feature will be fed into the network
    # Note that a feature can be used at multiple feed locations
    flc = [[] for i in range(FeedLoc.LEN)]

    # 2022-01-12 I found that feeding via the front seemed to be best
    flc[FeedLoc.conv].append('ema')
    flc[FeedLoc.conv].append('dvg')
    flc[FeedLoc.conv].append('volume')
    flc[FeedLoc.conv].append('logDiff')
    flc[FeedLoc.conv].append('rsi')
    flc[FeedLoc.conv].append('vix')

    config['feedLoc'] = flc

    # ****************************
    # TRAINING

    config['epochs'] = 8 # Number of complete passes of the data (subject to early stopping)
    
    config['dropout'] = 0.2 # Dropout_rate of layer applied before each LSTM or Conv1D. Set to 0 to disable

    # After training, change the weights to the model that had the best 
    # validation score
    config['revertToBest'] = True
    
    config['earlyStopping'] = 0 # 0 = no early stopping. Otherwise, the number
#     of epochs of no improvements before training is stopped (patience)
    
    # Optimizer
    config['optimiser'] = 'adam'

    config['learningRate'] = 0.002 # initial


    # ****************************
    # OTHER

    config['plotWidth'] = 9

    return config