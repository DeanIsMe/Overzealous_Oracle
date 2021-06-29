# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:19:35 2017

@author: Dean
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas

#==========================================================================
def PlotOutData(r, prices, output, sample=0, tRange=0):
    # Plot output
    if type(tRange) == int:
        tInd = range(r.timesteps)
    else:
        tInd = range(tRange[0], tRange[1])
    outDim = output.shape[-1]            
    x = tInd
    plt.figure(1, figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(x,prices[sample, tInd])
    plt.title('Price of {}'.format(r.coinList[sample]))
    plt.subplot(2,1,2)
    lines = list(range(outDim))
    for i in range(outDim):
        lines[i], = plt.plot(x,output[sample,tInd,i], label='Out{0}({1})'.format(i, r.config['outputRanges'][i]))
    l0, = plt.plot([x[0], x[-1]], [0, 0]) # Add line @ zero
    plt.legend(handles = lines)
    plt.title('Output for: {}'.format(r.coinList[sample]))
    plt.show()
    
    #==========================================================================
def PlotInData(r, dfs, sample=0, tRange=0):
    # Plot output
    if type(tRange) == int:
        tInd = np.array(range(r.timesteps))
    else:
        tInd = np.array(range(tRange[0], tRange[1]))   
    x = tInd    
    plt.figure(1, figsize=(20,6))
    plt.title('Inputs Data for {}'.format(r.coinList[sample]))
    df = dfs[sample]
    lines = list(range(len(df.columns)))
    for i, col in enumerate(df.columns):
        lines[i], = plt.plot(x, df[col][tInd], label=col)
    l0, = plt.plot([x[0], x[-1]], [0, 0]) # Add line @ zero
    plt.legend(handles = lines)
    plt.title('Input Data for: {}'.format(r.coinList[sample]))
    plt.show()

#==========================================================================
def CalcFavScores(config, prices):
    """
    prices should have shape (stocks, timesteps)
    out will have shape (stocks, timesteps, numPeriods)
    The number of periods is defined by the length of config['outputRanges']
    Can use this to output short, medium and long term scores
    """
    print('CalcFavScores Start')
    # If it's just a single row, add an extra dimension
    if (prices.ndim != 2):
        prices = prices.reshape(1,prices.shape[-1])
    
#    outPer = config['outputPeriods']
#    outDim = len(outPer)
#    pos = 1
#    scoreRanges = list()
#    for i in range(outDim):
#        scoreRanges.append((pos, outPer[i]))
#        pos = outPer[i]
        
    scoreRanges = config['outputRanges']
    outDim = len(scoreRanges)
        
    out = np.zeros((list(prices.shape)+[outDim]), dtype=float)
    for i, sRange in enumerate(scoreRanges):
        out[:,:,i] = CalcFavourabilityScore(config, prices, sRange)

    return out

    

#==============================================================================
def TranslateVector(input, translation, padding=0):
    """
    # Translate a vector to the left (neg) or right (pos)
    # Padding is a single value, that the vector will be padded with
    # Set padding==-2 to use the closest value
    # Output has same length as input (some padding values added, some removed)
    # Returns (output, valid)
    """
    if (padding == -2):
        # Use the closest value as the padding
        if (translation > 0):
            padding = input[:,0,None]
        else:
            padding = input[:,-1,None]
    
    len = input.shape[-1]
    output = np.ones(input.shape) * padding
    valid = np.zeros(input.shape, dtype=int)
    
    toCut = min(len, abs(translation))
    topIdx = len-toCut
    botIdx = toCut
    if translation >= 0: # Move to the right
        output[:,botIdx:] = input[:,:topIdx]
        valid[:,botIdx:] = 1
    elif translation < 0: # Move to the left
        output[:,:topIdx] = input[:,botIdx:]
        valid[:,:topIdx] = 1
    return (output, valid)

#==============================================================================
def CalcFavourabilityScore(config, price_Data, tRange):
    """ 
    price_Data is a numpy array of shape (samples, timesteps)
    price_Data should have each item corresponding to 1 trading day
    It should have no gaps, and should be adjusted for inflation
    It should be linear (not logarithmic)
    """
    
    # Generate the days at which I want to check the future prices
    numPoints = min(tRange[1]-tRange[0], 80)
    padding = 0
    
    b = 2 # in y = a*exp(b*x) + c. Affects linearity
    a = (tRange[1]-1) / (np.exp(b)-1)
    c = 1-a
    
    seedBot = np.log((tRange[0]-c)/a)/b
    seeds = np.linspace(seedBot,1,numPoints)
    checkDays = np.floor(a*np.exp(seeds*b)+c).astype(int)
    
    score = np.zeros(price_Data.shape)
    validity = np.zeros(price_Data.shape)
    
    # Get all of the future prices - positives
    for dayIdx in checkDays:
        [thisVec,  thisValid] = TranslateVector(price_Data,-dayIdx, padding)
        score = score + thisVec
        validity = validity + thisValid
    
    # Account for all of the subtractions of the current prices
    score = score - price_Data * validity
    # Normalise
    if config['pullUncertainYTo0']:
        score = (score / numPoints) / price_Data
    else:
        score = (score / validity) / price_Data
        

    # There will be data at the end that is NaN/Inf
    # Extend the last valid score out to the end
    score[:,-tRange[0]:] = score[:,-tRange[0]-1:-tRange[0]]
    
    if config['binarise']:
        # Minimise outliers reduces the size of outliers
        # Transform towards BINARY output
        # The minimiseOutliers =0.1 gives almost no reduction
        # =1.0 gives a big reduction of outliers
        # =5.0 is pretty much binary
        median = np.median(np.abs(score))
        mult = median / config['binarise']
        score = np.tanh(score/mult) * mult
        
    elif config['ternarise']:
        # Transform towards  buy/sell/neutral test
        # minimiseoutliers = 10 (strict ternary).
        # = 0.5 is less strict, but approaches binary
        # 1-3 seems preferable
        # selectivity 1=frequent buy/sell signals. 3=very picky buy/sell
        ind = score>0
        side = score[ind]
        median = np.median(side)
        score[ind] = (np.tanh((side/median-config['selectivity'])*config['ternarise']) + 1) * median
        
        ind = score<0
        side = score[ind]
        median = np.median(side)
        score[ind] = (np.tanh((side/median-config['selectivity'])*config['ternarise']) + 1) * median
    
    
    return score

#==========================================================================
def Normalise(dfs, cols):
    """
    dfs is a list of data frames
    cols is a list of the column names that need to be normalised
    All values in all dataframes are normalised by the same ratio
    """
    if not isinstance(cols, list):
        cols = [cols]
    
    # Determine the average of the 90th quantiles
    vals = []
    for df in dfs:
        for col in cols:
            vals.append(df[col].abs().quantile(0.90))
    
    # Get the scaler
    scaler = 1 / np.mean(vals)
    
    # Apply the scaler
    for df in dfs:
        for col in cols:
            df[col] *= scaler
    

#==========================================================================
def AddVix(r, dfs, prices):
    """
    # Add VIX - Volatility Index to a list of data frames
    """
    volatility = CalcVolatility(r.config, prices)
    for i in np.arange(r.sampleCount):
        for j in np.arange(volatility.shape[-1]):
            cName = 'vix{}'.format(j+1) # Column name
            dfs[i][cName] = volatility[i,:,j]
    
    # Volatility Normaliser
    # Normalise all by the same amount
    vixCols = [col for col in dfs[0].columns if 'vix' in col]
    Normalise(dfs, vixCols)
            
            
#==========================================================================
def CalcVolatility(config, prices):
    """
    # Volatility (erraticism)
    # Use an integral of the differential squared to measure this
    # Measure it over a large period, and also measure the comparative
    # erraticism of different time periods
    """
    
    # Absolute Erraticism Score
    samples = prices.shape[-2]
    timesteps = prices.shape[-1]
    
    # Comparative Erraticism
    # Set up the time periods
    numPeriods = config['vixNumPastRanges']
    maxDaysPast = config['vixMaxPeriodPast']
    firstPeriodLen = 5
    seeds = np.linspace(0,1,numPeriods)
    b = 3 # in y = a*exp(b*x) + c. Affects linearity. 4 is good. Lower: more linear
    a = (maxDaysPast-firstPeriodLen) / (np.exp(b)-1)
    c = firstPeriodLen-a
    periodStartsB4Today = np.round(a*np.exp(seeds*b)+c).astype(int)
    periodLengths = np.concatenate(([firstPeriodLen], np.diff(periodStartsB4Today))).astype(int)
    
    # Volatility
    # Diferences in the log2 domain are ratios, so the result is scale-agnostic (no further normalisation needed)
    noise = np.concatenate((np.square(np.diff(np.log2(prices))), [[0]]*samples), axis=1)
    volatility = np.zeros((samples, timesteps, numPeriods))
    validity = np.zeros((samples, timesteps, numPeriods))
    for periodNum in range(numPeriods):
        for subPointNum in range(periodLengths[periodNum]):
            [thisVec,  thisValid] = TranslateVector(noise, periodStartsB4Today[periodNum]-subPointNum, 0)
            volatility[:,:,periodNum] += thisVec
            validity[:,:,periodNum] += thisValid
    
    volatility = volatility/validity
    
    volatility[np.isinf(volatility)] = 0 # If there were no valid points, assume 0 noise.
    volatility[np.isnan(volatility)] = 0
    
    return volatility

#==========================================================================
def AddRsi(r, dfs):
    """
    RSI (relative strength index)
    This is essentially the RSI (exponential method), but represented differently
    Real RSI = 100.0 - (100.0 / (1.0 + RS)). It ranges from 0-100 (human intuitive)
    I calculate a number that hovers around -1 to 1 - machine intuitive
    dfs is a list fo data frames; a data frame for each sample
    """
    def CalcRSI(ser):
        # ser is a data series
        # Get the difference in price from previous step
        delta = ser.diff()
        # Get rid of the first row, which is NaN since it did not have a previous 
        # row to calculate the differences
        delta = delta[1:]
        
        # Make the positive gains (up) and negative gains (down) Series
        deltaUp, deltaDown = delta.copy(), delta.copy()
        deltaUp[deltaUp < 0] = 0
        deltaDown[deltaDown > 0] = 0
        
        # Exponential weighted moving average
        # Calculate the EWMA
        # 2 lines below are deprecated
#        roll_up1 = pandas.stats.moments.ewma(up, span=r.config['rsiWindowLen'])
#        roll_down1 = pandas.stats.moments.ewma(down.abs(), span=r.config['rsiWindowLen'])
        roll_up1 = deltaUp.ewm(span=r.config['rsiWindowLen'],min_periods=0,adjust=True,ignore_na=True).mean()
        roll_down1 = deltaDown.abs().ewm(span=r.config['rsiWindowLen'],min_periods=0,adjust=True,ignore_na=True).mean()
        
        # Calculate the RSI based on EWMA
        RSI = roll_up1 / roll_down1
        return np.log(RSI)
    
    for df in dfs:
        df['RSI'] = CalcRSI(df['close'])
        df['RSI'] = df['RSI'].replace([np.inf, -np.inf, np.nan], [2, -2, 0])
    
    Normalise(dfs, ['RSI'])
    
    return

#==========================================================================
def AddEma(r, dfs):
    """
    Add Exponential Moving Averages to the data frames
    """
    for df in dfs:
        for i, span in enumerate(r.config['emaLengths']):
            col = 'ema{}'.format(i+1)
            df[col] = df['close'].ewm(span=span,min_periods=0,adjust=True,ignore_na=False).mean() / df['close'] - 1

    emaCols = [col for col in dfs[0].columns if 'ema' in col]
    Normalise(dfs, emaCols)
    return

#==========================================================================
def AddLogDiff(r, dfs):
    """
    Add a logarithmic differential series to the data frames
    """
    for df in dfs:
        df['logDiff'] = df.close.apply(np.log2).diff()
        df['logDiff'] = df['logDiff'].replace([np.inf, -np.inf, np.nan], [2, -2, 0])

    Normalise(dfs, ['logDiff'])
    
    return

#==============================================================================
def ScaleLoadedData(dfs):
    """
    # dfs must be a list of DataFrames
    # Each DataFrame input should have 4 columns: close, high, low, volume
    # This functions prepares the data for entry into a neural network
    """
    
    ##22/10/2017: normalising each input sequence was shown to be best
    #
    #Scale values to a reasonable range
    #17/12/2017: dividing by 90th percentile was found to be a good scale
    
    # Functions of data
    for df in dfs:
        # Make high and low relative to the close
        df.high = (df.high / df.close - 1)
        df.low = (df.low / df.close - 1)
    
    # Normalisation per stock
    for df in dfs:
        # Normalise the close value
        df.close /= df.close.quantile(0.9)
        # Normalise the volume
        df.volume /= df.volume.quantile(0.9)
    
    # Normalisation high and low equally over all stocks
    Normalise(dfs, ['high', 'low'])
           
    return