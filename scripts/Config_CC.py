# -*- coding: utf-8 -*-
# Cryptocurrency configuration

from DataTypes import FeedLoc

#==============================================================================
# GetConfig
# Defines and returns the configuration dict
def GetConfig():
    config = dict()
    
    # ****************************
    # INPUT DATA
    # Volatility
    config['vixNumPastRanges'] = 3 # number of ranges to use
    config['vixMaxPeriodPast'] = 24*14
    
    # RSI - Relative Strength Index
    config['rsiWindowLen'] = 96 # The span of the EMA calc for RSI
    
    # Exponential Moving Average
    config['emaLengths'] = [24, 168, 400] # The span of the EMAs



    # ****************************
    # OUTPUT (TARGET) DATA

    # Favourability Score: CompareToFutureData
    # How many days to exclude from training because the 'to buy' score is not well defined
    config['excludeRecentDays'] = 50 # Tradeoff between how recent, and accuracy to 'to Buy' score
    
    
    # config['outputRanges'] = [[1,5], [6,25], [26,125]] # The ranges over which to calculate output scores
    # Starts at 1
    # I'm using hourly data
    config['outputRanges'] = [[26,125]]
    # Also defines the number of output periods
    
    # When the date is within 'excludeRecentDays' of the end of the data, then
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
    config['binarise'] = 0.2
    
    # Ternarise
    # Transform towards a ternary buy/sell/neutral
    # minimiseoutliers = 10 (strict ternary). 
    # = 0.5 is less strict, but approaches binary
    # 1-3 seems preferable
    # Binarise and ternarse must not both be non-zero
    config['ternarise'] = 0

    # selectivity 1=frequent buy/sell signals. 3=very picky buy/sell
    config['selectivity'] = 2
    # A good balance seems to be ternarise = 1, selectivity = 2


    # ****************************
    # NEURAL NET (MODEL)

    config['bottleneckWidth'] = 128 # A dense layer is added before the LSTM to reduce the LSTM size
    config['lstmWidths'] = [128] # Number of neurons in each LSTM layer. They're cascaded.
    # The 3 parameters below can be a list (one value per conv layer), or a scalar (apply to all conv layers)
    # The num of conv layers will be the greatest number of valid layers
    # If any parameter is empty ([] or 0), then there will be no convolutional layers
    config['convDilation'] = [1,2,4,8,16,32,64,128] # Time dilation factors. 
    config['convFilters'] = [80,75,70,65,60,50,40,30] # Number of filters per layer. List or scalar
    config['convKernelSz'] = 10 # Kernel size per filter



    # ****************************
    # TRAINING DATA

    config['dataRatios'] = [0.75, 0.2 ,0.05] # Training, Validation, Testing
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

    flc[FeedLoc.dense].append('high')
    flc[FeedLoc.dense].append('low')
    flc[FeedLoc.lstm].append('volume')
    flc[FeedLoc.conv].append('close')
    flc[FeedLoc.conv].append('ema')
    flc[FeedLoc.lstm].append('logDiff')
    flc[FeedLoc.dense].append('logDiff')
    flc[FeedLoc.dense].append('vix')
    flc[FeedLoc.lstm].append('RSI')

    config['feedLoc'] = flc

    # ****************************
    # TRAINING

    config['epochs'] = 8 # Number of complete passes of the data (subject to early stopping)
    

    config['dropout'] = 0.1 # Dropout_rate of layer applied before each LSTM or Conv1D. Set to 0 to disable

    # After training, change the weights to the model that had the best 
    # validation score
    config['revertToBest'] = True
    
    config['earlyStopping'] = 0 # 0 = no early stopping. Otherwise, the number
#     of epochs of no improvements before training is stopped (patience)
    
    # Optimizer
    config['optimiser'] = 'adam'


    return config