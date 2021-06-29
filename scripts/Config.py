# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:22:48 2017

@author: Dean
"""

#==============================================================================
# GetConfig
# Defines and returns the configuration dict
def GetConfig():
    config = dict()
    
    # Favourability Score: CompareToFutureData
    # How many days to exclude from training because the 'to buy' score is not well defined
    config['excludeRecentDays'] = 30 # Tradeoff between how recent, and accuracy to 'to Buy' score
    
#    config['outputRanges'] = [[1,5], [6,25], [26,125]] # The ranges over which to calculate output scores
    config['outputRanges'] = [[1,100]]
    # Also defines the number of output periods
    # About 246 trading days per year
    
    # When the date is within 'excludeRecentDays' of the end of the data, then
    # the 'favourability' score cannot be completely calculated.
    # When pullUncertainYTo0 is true, the favourability score will be pulled
    # more towards zero as the more and more dates are missing. When it's
    # false, the score will be calculated assuming that the price does not vary
    # at all from the last day that data exists
    config['pullUncertainYTo0'] = True
    
    
#    # Volume Analysis
#    config['volMaxDaysPast'] = 246
#    
#    # Exchange data
#    # TWI: trade weighted index
#    config.exchList = {'TWI', 'USD', 'CNY'}
#    config.exchNumPastPeriods = 3
#    config.exchMaxDaysPast = 300
#        
#    # Neural Network
    config['neurons'] = [64] # Number of neurons in LSTM
    config['epochs'] = 64 # Number of complete passes of the data
#    config.neuronsFactor = 0.8 # Size of the hidden layer
#    config.divideFcn = 'dividerand' # divideIntBlocks or dividerand
#    config.valBlockSize = 5 # Only applies to divideIntBlocks
    config['dataRatios'] = (0.7, 0.15, 0.15) # Training, Validation, Testing
    
    return config