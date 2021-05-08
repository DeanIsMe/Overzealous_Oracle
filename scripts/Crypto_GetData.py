# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:32:54 2018

@author: Dean
"""
import numpy as np
import pandas as pd
import crycompare as cc
import time

#==========================================================================
def GetHourlyDf(coins, numHours):
    """
    Returns a list of dataframes
    Columns in output are close, high, low
    """
    tMax = 80*24*60*60 # Seconds. Max of 2000 points, so I'll do only 80 days per call (1920 hours)
    dfs = []
    tNow = time.time()
    for coin in coins:
        print('Getting data for {}'.format(coin))
        
        t = tNow - numHours*60*60 # unix time (second since 1970)
        success = True
        df = pd.DataFrame()
        while t < tNow:
            timeSpan = min(tMax, tNow - t)
            t += timeSpan
            print('Getting {:4.0f} points, up to time {}'.format(timeSpan/3600, time.ctime(t)))
            histo = cc.History().histoHour(coin, 'USD', limit=timeSpan/3600-1, toTs = t)
            
            if (histo['Data']):
                thisDf = pd.DataFrame(histo['Data'])
                thisDf['time'] = pd.to_datetime(thisDf['time'],unit='s')
                thisDf.index = thisDf['time']
                thisDf.pop('time')
                thisDf.pop('volumefrom') # keep only USD volume (volumeto)
                thisDf.pop('open')
                # Replace 'volumeto' with 'volume'
                thisDf.rename(columns={'volumeto':'volume'},  inplace=True)
                if df.empty:
                    df = thisDf
                else:
                    df = pd.concat([df, thisDf])
                
            else:
                print('\nCOIN NOT FOUND! ' + coin);
                success = False
                break
        
        if success:
            dfs.append(df)
    # Prune all arrays to have the same max length !@#
    return dfs

# Test
#dfs = GetHourlyDf(['ETH'], 25)