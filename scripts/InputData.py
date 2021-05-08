# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:24:56 2017

@author: Dean
"""

import numpy as np

def NormaliseEachSeq(inSeq):
    return inSeq / np.max(inSeq, axis=1).reshape(inSeq.shape[0],1)

def LogDiffSeries(inSeq):
    """Convert some input data to a difference series in the log2 domain"""
    out = np.diff(np.log2(inSeq), axis=1)
    #There's no 'diff' for the first day. Add it as zero, so that dates correlate
    out = np.concatenate((np.zeros((inSeq.shape[0], 1)), out), axis=1)
    return out

def LogSeries(inSeq):
    """Convert some input data to a normalised log series in the log2 domain"""
    return NormaliseEachSeq(np.log2(inSeq))

