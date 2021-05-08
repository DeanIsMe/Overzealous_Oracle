# -*- coding: utf-8 -*-
# Cryptocurrency configuration

#==============================================================================
# GetConfig
# Defines and returns the configuration dict
def GetConfig():
    config = dict();
    
    # Favourability Score: CompareToFutureData
    # How many days to exclude from training because the 'to buy' score is not well defined
    config['excludeRecentDays'] = 50 # Tradeoff between how recent, and accuracy to 'to Buy' score
    
    
#    config['outputRanges'] = [[1,5], [6,25], [26,125]] # The ranges over which to calculate output scores
    # Starts at 1
    config['outputRanges'] = [[12,48]]
    # Also defines the number of output periods
    # About 246 trading days per year
    
    # When the date is within 'excludeRecentDays' of the end of the data, then
    # the 'favourability' score cannot be completely calculated.
    # When pullUncertainYTo0 is true, the favourability score will be pulled
    # more towards zero as the more and more dates are missing. When it's
    # false, the score will be calculated assuming that the price does not vary
    # at all from the last day that data exists
    config['pullUncertainYTo0'] = True;
    
    
#    # Volume Analysis
#    config['volMaxDaysPast'] = 246;
    
    # After training, change the weights to the model that had the best 
    # validation score
    config['revertToBest'] = True
    
    config['earlyStopping'] = 15 # 0 = no early stopping. Otherwise, the number
#     of epochs of no improvements before training is stopped
    
    config['dropout'] = 0.25 # Dropout layer applied at inputs of each LSTM       
#    
#        
#    # Neural Network
    config['neurons'] = [128, 128]; # Number of neurons in LSTM
    config['epochs'] = 64; # Number of complete passes of the data (subject to early stopping)

    config['dataRatios'] = [0.75, 0.2 ,0.05] # Training, Validation, Testing
    #note: 03/02/2018 I am using the same set for validation and testing
    # I'm just using the 'test' set to generate state for prediction
    
    # OUTPUT DATA TRANSFORMATION
    # (only 1 can be non-zero)
    
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
    config['ternarise'] = 0
    # selectivity 1=frequent buy/sell signals. 3=very picky buy/sell
    config['selectivity'] = 2
    # A good balance seems to be ternarise = 1, selectivity = 2
    
    # Volatility
    config['vixNumPastRanges'] = 3 # number of ranges to use
    config['vixMaxPeriodPast'] = 24*14
    
    # RSI - Relative Strength Index
    config['rsiWindowLen'] = 96 # The span of the EMA calc for RSI
    
    # Exponential Moving Average
    config['emaLengths'] = [24, 168, 400] # The span of the EMAs
    
    # Input data and output data are divided by 90th percentile. Then, they
    # are multiplied by the 'scale'. This has a massive impact on the training
    # and results
    config['inScale'] = 1
    config['outScale'] = 1
    
    return config