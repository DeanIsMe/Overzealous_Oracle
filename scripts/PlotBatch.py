# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:50:09 2017
Updated 2022

@author: Dean
"""

#%%
# A GRID OF PLOTS

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numbers
from NeuralNet import PlotTrainMetrics

# Uncomment either option A or B
# OPTION A: batVal1 along x (columns), batVal2 along y (rows)
#plt.figure(figsize=(bat1Len*5,bat2Len*3)); p = 1
#for idx2 in range(bat2Len):
#    for idx1 in range(bat1Len):
#        plt.subplot(bat2Len, bat1Len, p)
        
# OPTION B: batVal2 along x (columns), batVal1 along y (rows)
fig, axs = plt.subplots(bat1Len, bat2Len, figsize=(bat2Len*5,bat1Len*3)); p = 1
fig.tight_layout()
minY = 9e9
maxY = -9e9
for idx1 in range(bat1Len):
    rowAx = []
    for idx2 in range(bat2Len):
        ax = axs[idx1, idx2]
        r = results[idx2][idx1] # Pointer for brevity
        
        (thisMaxY, thisMinY) = PlotTrainMetrics(r, ax)
        maxY = max(maxY, thisMaxY)
        minY = min(minY, thisMinY)
        
        ax.set_title(f'{bat2Name}:{bat2Val[idx2]}, {bat1Name}:{bat1Val[idx1]}', fontdict={'fontsize':10})
        #ax.set_yscale('log')
        ax.grid()
        
        print('{}:{}, {}:{}'.format(bat2Name, bat2Val[idx2], bat1Name, bat1Val[idx1]))
        print('Train Score: {:5}\nTest Score: {:5} (1=neutral)'.format(r.trainScore, r.testScore))

maxY = round(maxY+0.05, 1)
minY = round(minY-0.05, 1)
# Set all to have the same axes
for idx1 in range(bat1Len):
    for idx2 in range(bat2Len):
        axs[idx1,idx2].set_ylim(bottom=minY, top=maxY)
        axs[idx1,idx2].set_xlim(left=0, right=r.config['epochs']-1)
plt.show()

#%%
# Plot each value against score

# 1 PLOT, MULTIPLE LINES
def DrawPlot(valA, valB, nameA, nameB, data, nameY):
    # valA is the x axis
    if (not isinstance(valA[0], numbers.Number) or len(valA) < 3):
        return
    fig, ax = plt.subplots(figsize=(7,4))
    fig.tight_layout()
    ax.plot(valA, data)
    diffA = np.diff(valA)
    if diffA[-1]/diffA[0] > 5:
        ax.set_xscale('log')
        ax.set_xticks(valA)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(nameA)
    ax.set_ylabel(nameY)
    ax.set_title('{} vs {} (Legend = {})'.format(nameY, nameA, nameB))
    ax.legend(valB)
    plt.show()

# Test Score vs bat1Val
data = np.array([[r.testScore for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Test Score')

# Test Score vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Test Score')

# Train Score vs bat1Val
data = np.array([[r.trainScore for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Train Score')

# Train Score vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Train Score')

# Training Time vs bat1Val
data = np.array([[r.trainTime for r in results[idx2]] for idx2 in range(bat2Len)])
DrawPlot(bat1Val, bat2Val, bat1Name, bat2Name, data.transpose(), 'Training Time')

# Training Time vs bat2Val
DrawPlot(bat2Val, bat1Val, bat2Name, bat1Name, data, 'Training Time')

## PLOT ALL PREDICTIONS
#for idx1 in range(bat1Len):
#    for idx2 in range(bat2Len):
#        r = results[idx2][idx1] # Pointer for brevity
#        TestNetwork(r, prices, thisInData, thisOutData, tInd)
# %%
