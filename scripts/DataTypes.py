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
    rnn = 1
    dense = 0
    null = -1
    LEN = 3
    NAMES = {conv:'conv', rnn:'rnn', dense:'dense', null:'null'}
    

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
        self.neutralValAbsErr = None
        self.neutralValSqErr = None
        self.trainScore = None # neutralTrainAbsErr / trainAbsErr
        self.testScore = None # neutralTestAbsErr / testAbsErr

        self.modelEpoch = -1 # The most recent epoch trained for the current model weights.
        # epochs are 0-numbered, so total epochs trained is modelEpoch+1
        
        self.testAbsErr = None
        self.neutralTestAbsErr = None # test error if output was 'always neutral'
        
        
from IPython.display import Markdown, display
def printmd(string, color=None):
    if color is None:
        display(Markdown(string))
    else:
        colorstr = "<span style='color:{}'>{}</span>".format(color, string)
        display(Markdown(colorstr))


def SecToHMS(t):
    """Makes a string like ' 2:15:36' to represent some duration, given in seconds. 8 chars"""
    return f"{t//3600.:2.0f}:{(t%3600)//60.:02.0f}:{t%60.:02.0f}"