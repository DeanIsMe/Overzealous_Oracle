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

class TrainMetrics():
    # Saves metrics each epoch as the model trains
    def __init__(self):
        # Capturing Training
        self.lossTrain = []
        self.absErrTrain = []
        self.lossVal = []
        self.absErrVal = []
        self.fitness = []
        self.curEpoch = 0 # Counted from the start of this model creation,
        
        
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
    LIST = [conv, lstm, dense]
    

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
               
        self.trainMetrics = 0 # See class above
        self.coinList = []

        self.inFeatureList = None
        self.feedLocFeatures = None
        self.tInd = None # Dictionary with keys: 'train', 'val', 'test'. Values are the time indices in each set
        
        self.model = None
        self.kerasOpt = 0 # 0 = Adam. Otherwise, set to an Optimizer
        self.kerasOptStr = None
        self.config = None
        
        self.date = datetime.date.today()
        self.trainTime = None
        self.prediction = []
        self.trainAbsErr = None
        self.neutralTrainAbsErr = None # train error if output was 'always neutral'
        self.neutralTrainSqErr = None # train error if output was 'always neutral'
        self.trainScore = None # neutralTrainAbsErr / trainAbsErr
        
        self.neutralValAbsErr = None
        self.neutralValSqErr = None
        
        self.testAbsErr = None
        self.neutralTestAbsErr = None # test error if output was 'always neutral'
        self.testScore = None # neutralTestAbsErr / testAbsErr
        
