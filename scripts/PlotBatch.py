# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:50:09 2017

@author: Dean
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numbers
from NeuralNet import PlotTrainData


# A GRID OF PLOTS

# Uncomment either option A or B
# OPTION A: batVal1 along x (columns), batVal2 along y (rows)
#plt.figure(1, figsize=(bat1Len*5,bat2Len*3)); p = 1
#for idx2 in range(bat2Len):
#    for idx1 in range(bat1Len):
#        plt.subplot(bat2Len, bat1Len, p)
        
# OPTION B: batVal2 along x (columns), batVal1 along y (rows)
plt.figure(1, figsize=(bat2Len*5,bat1Len*3)); p = 1
minY = 9e9
maxY = -9e9
allAx = []
for idx1 in range(bat1Len):
    rowAx = []
    for idx2 in range(bat2Len):
        plt.subplot(bat1Len, bat2Len, p)
        
        r = results[idx2][idx1] # Pointer for brevity
        # As 'p' increases, plot starts top-left and moves to the right
        ax = plt.gca()
        rowAx.append(ax)
        
        (thisMaxY, thisMinY) = PlotTrainData(r, True)
        maxY = max(maxY, thisMaxY)
        minY = min(minY, thisMinY)
        
        plt.title('{}:{}, {}:{}'.format(bat2Name, bat2Val[idx2], bat1Name, bat1Val[idx1]))
        ax.set_yscale('log')
        
        p += 1
        print('{}:{}, {}:{}'.format(bat2Name, bat2Val[idx2], bat1Name, bat1Val[idx1]))
        print('Train Score: {:5}\nTest Score: {:5} (1=neutral)'.format(r.trainScore, r.testScore))
    allAx.append(rowAx)
    
for idx1 in range(bat1Len):
    for idx2 in range(bat2Len):
        allAx[idx1][idx2].set_ylim(bottom=max(minY,0.1), top=maxY)
        allAx[idx1][idx2].set_xlim(left=0, right=r.config['epochs'])
plt.show()

# 1 PLOT, MULTIPLE LINES
def DrawPlot(valA, valB, nameA, nameB, data, nameY):
    # valA is the x axis
    if (not isinstance(valA[0], numbers.Number) or len(valA) < 3):
        return
    plt.figure(2, figsize=(12,6));
    ax = plt.gca()
    ax.plot(valA, data);
    diffA = np.diff(valA)
    if diffA[-1]/diffA[0] > 5:
        ax.set_xscale('log')
        ax.set_xticks(valA)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel(nameA);
    plt.ylabel(nameY);
    plt.title('{} vs {} (Legend = {})'.format(nameY, nameA, nameB))
    plt.legend(valB); 
    plt.show()

# Test Error vs bat1Val
data = np.array([[r.testScore for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Test Score');

# Test Error vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Test Score');

# Train Error vs bat1Val
data = np.array([[r.trainScore for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Train Score');

# Train Error vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Train Score');

# Training Time vs bat1Val
data = np.array([[r.trainTime for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Training Time');

# Training Time vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Training Time');

## PLOT ALL PREDICTIONS
#for idx1 in range(bat1Len):
#    for idx2 in range(bat2Len):
#        r = results[idx2][idx1] # Pointer for brevity
#        TestNetwork(r, prices, thisInData, thisOutData, tInd)