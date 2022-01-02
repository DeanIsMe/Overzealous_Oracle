# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 10:47:44 2018

@author: Dean

This file contains datatypes
It's useful to have them in a separate file for picling purposes
(If a file changes, then any classes within that file are considered
different)
"""

import datetime
       
class FeedLoc():
    """
    ## FeedLoc is an enum
    The variable represents the location where the feature will be fed into the network
    The value is the index in the list of inputs
    """
    conv = 2
    lstm = 1
    dense = 0
    null = -1
    LEN = 3
    NAMES = {conv:'conv', lstm:'lstm', dense:'dense', null:'null'}
    

class ModelResult():
    # Saves all data about a model, including training and testing
    # Used to save information about batches
    def __init__(self):
        self.isBatch = False
        self.batchName = ''
        self.batchRunName = ''
        self.sampleCount = 0
        self.timesteps = 0
        self.inFeatureCount = 0
        self.outFeatureCount = 0
               
        self.trainHistory = {} # hist.history, as returned by model.fit()
        self.coinList = []
        self.numHours = 0

        self.inFeatureList = None
        self.feedLocFeatures = None
        self.tInd = None # Dictionary with keys: 'train', 'val', 'test'. Values are the time indices in each set
        
        self.model = None
        self.config = None
        
        self.date = datetime.date.today()
        self.trainTime = None
        self.prediction = []
        self.trainAbsErr = None
        self.neutralTrainAbsErr = None # train error if output was 'always neutral'
        self.neutralTrainSqErr = None # train error if output was 'always neutral'
        self.trainScore = None # neutralTrainAbsErr / trainAbsErr
        self.modelEpoch = 0 # The epochs trained for the current model weights
        
        self.neutralValAbsErr = None
        self.neutralValSqErr = None
        
        self.testAbsErr = None
        self.neutralTestAbsErr = None # test error if output was 'always neutral'
        self.testScore = None # neutralTestAbsErr / testAbsErr
        
