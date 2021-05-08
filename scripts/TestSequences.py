# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 21:43:59 2017

@author: Dean
"""

import numpy as np
import matplotlib.pyplot as plt

testLen = 1000
noiseLevel = 0.01
sinInput4 = np.zeros((3,testLen))
rd = np.random.normal(1, noiseLevel, (testLen))
sinInput4[0] = ((np.sin(np.linspace(0, 10*np.pi, testLen))+5)*rd).reshape((testLen))
rd = np.random.normal(1, noiseLevel, (testLen))
sinInput4[1] = ((np.sin(np.linspace(0, 18*np.pi, testLen)+1)+10)*rd).reshape((testLen))*2
rd = np.random.normal(1, noiseLevel, (testLen))
sinInput4[2] = ((np.sin(np.linspace(0, 30*np.pi, testLen)-1)+21)*rd).reshape((testLen))*3


if __name__ == "__main__":
    inSeq = sinInput4
    for i in range(inSeq.shape[0]):
        plt.plot(inSeq[i,:])
    plt.show()

def GetInSeq():
    return sinInput4