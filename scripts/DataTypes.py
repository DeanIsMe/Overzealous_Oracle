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

class TrainData():
    # Saves data each epoch as the model trains
    def __init__(self):
        # Capturing Training
        self.lossTrain = []
        self.absErrTrain = []
        self.lossVal = []
        self.absErrVal = []
        self.fitness = []
        self.curEpoch = 0 # Counted from the start of this model creation,
        
        

class ModelResult():
    # Saves all data about a model, including training and testing
    # Used to save information about batches
    def __init__(self):
        self.isBatch = False;
        self.batchName = '';
        self.batchRunName = '';
        self.samples = 0;
        self.timesteps = 0;
        self.inFeatures = 0;
        self.outFeatures = 0;
               
        self.trainData = 0 # See class above
        self.inDataColumns = []
        self.coinList = []
        
        self.model = 0;
        self.kerasOpt = 0;
        self.kerasOptStr = ''
        self.modelSummary = ''
        self.config = 0;
        
        self.date = datetime.date.today()
        self.trainTime = 0;
        self.prediction = [];
        self.trainAbsErr = 0
        self.neutralTrainAbsErr = 0 # train error if output was 'always neutral'
        self.neutralTrainSqErr = 0 # train error if output was 'always neutral'
        self.trainScore = 0 # neutralTrainAbsErr / trainAbsErr
        
        self.neutralValAbsErr = 0
        self.neutralValSqErr = 0
        
        self.testAbsErr = 0
        self.neutralTestAbsErr = 0 # test error if output was 'always neutral'
        self.testScore = 0 # neutralTestAbsErr / testAbsErr
        
