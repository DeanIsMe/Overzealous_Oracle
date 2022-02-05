# -*- coding: utf-8 -*-
"""
Cryptocurrency correlation analysis
Created on Jan 29  2022
@author: Dean

How should I go about performing a correlation analysis?
I could find tokens that are normally correlated, for the purpose
of identifying when the correlation breaks (assuming it will later catch up).

Conversely, I could look for some kind of time-delayed correlation,
where 1 token moves AFTER another token does.
I chose the latter.


"""

#%% 
# IMPORTS & SETUP

# note that "matplotlib notebook" isn't working for me
%matplotlib widget

import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))
print(f'Working directory is "{os.getcwd()}"')

import numpy as np

from DataTypes import ModelResult, printmd, SecToHMS
from TestSequences import GetInSeq
import InputData as indata
import copy

import datetime

import matplotlib.pyplot as plt
import time
import Crypto_GetData as cgd
import pickle
from datetime import datetime
import pandas as pd


# Load the input data file
#At this point, the stock data should have all gaps filled in
if not 'dataLoader' in locals():
    inDataFileName = './indata/2021-12-31_price_data_60m.pickle'
    dataLoader = cgd.DataLoader(inDataFileName)
    print('Loaded input file')

class CorrConfig():
    def __init__(corrCfg):
        corrCfg.numHours = int(24 * 365 * 0.5)
        corrCfg.valNumHours = int(24 * 365 * 0.1)

corrCfg = CorrConfig()

# %%
# Collect data
#dfs = dataLoader.GetHourlyDf(r.config['coinList'], r.config['numHours']) # a list of data frames
pair_summary = dataLoader.GetPairsSummary()

# Remove all pairs that don't have up to date data
pair_summary = pair_summary[pair_summary.last_time == pair_summary.last_time.max()]

# Keep only pairs that are paired with USD
pair_summary = pair_summary[pair_summary['pair'].apply(lambda x: x[-3:] == 'usd' or x[-4:] == 'usdt')]

# Keep only pairs that have the desired duration available
totalHours = corrCfg.numHours + corrCfg.valNumHours
pair_summary = pair_summary[pair_summary['dur_avail'] >= totalHours * 3600]

# Get the data for these pairs
pair_list = list(pair_summary['pair'])

pair_list = [p for p in pair_list if 'usdc' not in p]

dfs = dataLoader.GetHourlyDfPairs(pair_list, totalHours) # a list of data frames

# Combine close data into single dict
price_dict = {}
price_dict_val = {}
for df in dfs:
    price_dict[df.name] = df['close'].values[:corrCfg.numHours]
    price_dict_val[df.name] = df['close'].values[-corrCfg.valNumHours:]

# %%
"""Correlation analysis
For every pair of coins, coinA and coinB,
find the correlation between:
    coinA's price movement over the previous X hours, and
    coinB'a price movement over the following Y hours

I always keep X=Y=steps_past
"""

def CalcCorr(price_dict, steps_past, avg_steps=3):

    #steps_past = 24
    # avg_steps: How many timesteps to average

    avg_steps = min(avg_steps, steps_past)

    offset = steps_past - avg_steps # <avg_steps> into the past are handled by the rolling window. Add this many extra steps to get to <max_steps_past>

    class Timer():
        def __init__(self):
            self.start_time = time.perf_counter()

        def start(self): # optional
            elapsed = self.elapsed()
        
        def elapsed(self):
            return time.perf_counter() - self.start_time
        
        def stop(self):
            s = self.elapsed()
            return s



    t1 = Timer()

    prices_smth = {}
    for pair in price_dict:
        # Use pandas average, because it supports a window that's right-aligned
        # whereas numpy's window is center aligned
        prices_smth[pair] = pd.Series(price_dict[pair]).rolling(window=avg_steps, min_periods=0, center=False).mean().values
    
    # Calculate divergence
    dvg = {}
    for pair in price_dict:
        dvg[pair] = price_dict[pair][steps_past:] / prices_smth[pair][:-steps_past] - 1

    # all together correlate method:
    dvg_2d = np.stack(list(dvg.values()), axis=1)

    corr = np.corrcoef(dvg_2d[steps_past:, :], dvg_2d[:-steps_past, :], rowvar=False)
    # The above performs a lot of unnecessary correlations. Remove them
    corr = corr[len(pair_list):, :len(pair_list)]

    if 0:
        # pandas method of calculating correlation coefficients
        # this is more processing-intensive and more complicated than numpy method,
        # because pandas tries to align rows
        prices_df = pd.DataFrame(price_dict)

        dvg = pd.DataFrame()
        for pair in prices_df.columns:
            prices_smth = prices_df[pair].rolling(window=avg_steps, min_periods=0).mean()
            dvg.loc[:,pair] = prices_df[pair].iloc[steps_past:].multiply(1 / prices_smth.values[:-steps_past])  - 1

        
        df_a = dvg.iloc[steps_past:,:]
        df_b = dvg.iloc[:-steps_past,:]
        df_a.reset_index(drop=True, inplace=True)
        df_b.reset_index(drop=True, inplace=True)
        df_b.columns = [col + f'_p{steps_past}' for col in df_b.columns]
        corr_df = pd.concat([df_a, df_b], axis=1).corr()

        # The above performs a lot of unnecessary correlations. Remove them
        corr_df = corr.loc[df_b.columns, df_a.columns]

    return corr

    

results_str=[]
results = {'steps_past':[], 'best_corr':[], 'val_corr':[], 'coin_A':[], 'coin_B':[]}
for steps_past in range(1,10):
    corr = CalcCorr(price_dict, steps_past)
    corr_val = CalcCorr(price_dict_val, steps_past)
    # Find the best results
    best_idx = np.unravel_index(np.argmax(corr), corr.shape)
    coin_a = pair_list[best_idx[0]]
    coin_b = pair_list[best_idx[1]]
    results['steps_past'].append(steps_past)
    results['best_corr'].append(corr[best_idx])
    results['val_corr'].append(corr_val[best_idx])
    results['coin_A'].append(coin_a)
    results['coin_B'].append(coin_b)

    #res_str = f"steps_past = {steps_past:5}. best corr = {corr[best_idx]:6.3f}.   {coin_a:8} & {coin_b:8}"
    #results_str.append(res_str)

    # Print data (Dataframe is better for printing)
    #corr_df = pd.DataFrame(corr, columns=pair_list, index=[p + f'_p{steps_past}' for p in pair_list])
    #print(f"numpy:{t1.stop()}")

results_df = pd.DataFrame(results)

print(f"Config: {corrCfg.numHours/24} days, validation {corrCfg.valNumHours/24} days")
print("Best of the best results:")
print(results_df.iloc[results_df['val_corr'].argmax(), :])
results_df
# #%%
# fig, ax = plt.subplots()
# pair = 'ethusd'
# ax.plot(prices_df[pair])
# ax2 = ax.twinx()
# ax2.plot(dvg[pair], color='orange')
# %%
# Plot the result with the best validation coefficient
best_steps_idx = np.argmax(results['val_corr'])
steps_past=results['steps_past'][best_steps_idx]
corr = CalcCorr(price_dict, steps_past)

# Find the best results
best_pair_idx = np.unravel_index(np.argmax(corr), corr.shape)
coin_a = pair_list[best_pair_idx[0]]
coin_b = pair_list[best_pair_idx[1]]

fig, ax = plt.subplots(figsize=(12, 5))
fig.tight_layout()

def Scale(arr):
    return arr / np.mean(arr)

lines = []
x0 = 1000
x1 = x0 + 500
lines += ax.plot(Scale(price_dict[coin_a][x0:x1]), label=coin_a)
lines += ax.plot(Scale(price_dict[coin_b][x0:x1]), label=coin_b)
ax.grid()
ax.legend(handles=lines)
ax.set_title(f"steps_past={steps_past}. val_corr={results['val_corr'][best_steps_idx]}")



# %% 
print('DONE')