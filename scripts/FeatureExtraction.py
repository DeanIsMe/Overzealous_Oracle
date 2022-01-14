# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:19:35 2017

@author: Dean
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame

# df: dataframe
# dfs: dataframes
# There is one dataframe per ticker/coin


from DataTypes import printmd
from scripts.DataTypes import FeedLoc

#==========================================================================
def PlotInData(r, dfs, sample=0, tRange=2000, colPatterns=[]):
    """
    Plots the input data for 1 sample (df)
    sample is the index of the sample to plot
    tRange Specifies the number of timesteps (or time range)) e.g. [2000, 4000]
    colPatterns specifies which features to plot. e.g. ['ema','rsi0_96']
        a partial string is sufficient to match
    """
    if type(tRange) == int:
        tInd = np.array(range(r.timesteps))
    else:
        tInd = np.array(range(tRange[0], tRange[1]))   
    x = tInd
    fig, ax = plt.subplots(figsize=(r.config['plotWidth'], 4))
    fig.tight_layout()
    df = dfs[sample]
    lines = []
    
    if isinstance(colPatterns, str): colPatterns = [colPatterns]
    if not colPatterns:
        # Generate colPatterns from config feedLoc
        colPatterns = set()
        for f in range(FeedLoc.LEN):
            for cp in r.config['feedLoc'][f]:
                colPatterns.add(cp)
    # Find all columns that match the pattern(s)
    columns=[]
    for col in df.columns:
        for cp in colPatterns:
            if cp in col:
                columns.append(col)
                break
    # Plot the columns
    for col in columns:
        lines += ax.plot(x, df[col][tInd], label=col)
    # Add price data
    lines += ax.plot(x, ShiftAndScaleCol(df['close'][tInd]), label='price')
    l0, = ax.plot([x[0], x[-1]], [0, 0]) # Add line @ zero
    ax.legend(handles = lines)
    ax.set_title('Input Data for: {}'.format(r.config['coinList'][sample]))
    ax.grid()
    #plt.show()

#==========================================================================
def PlotOutData(r, prices, output, sample=0, tRange=0):
    # Plot output
    if type(tRange) == int:
        tInd = range(r.timesteps)
    else:
        tInd = range(tRange[0], tRange[1])
    outDim = output.shape[-1]            
    x = tInd
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(r.config['plotWidth'] , 7))
    fig.tight_layout()
    ax1.plot(x,prices[sample, tInd])
    ax1.set_title('Price of {}'.format(r.config['coinList'][sample]))
    ax1.grid()
    lines = list(range(outDim))
    for i in range(outDim):
        lines[i], = ax2.plot(x,output[sample,tInd,i], label='Out{0}({1})'.format(i, r.config['outputRanges'][i]))
    l0, = ax2.plot([x[0], x[-1]], [0, 0]) # Add line @ zero
    ax2.legend(handles = lines)
    ax2.set_title('Output for: {}'.format(r.config['coinList'][sample]))
    ax2.grid()
    plt.show()

#==========================================================================
def PrintInOutDataRanges(dfs, outData):
    # Print a table. Each column is a feature, each row is a sample
    quantiles = [0.90, 0.10]
    values = []

    for i, q in enumerate(quantiles):
        series = []
        for df in dfs:
            # bool columns break quantile, so exclude them
            ser = df.loc[:,df.dtypes != bool].quantile(q=q)
            ser.name = df.name
            series.append(ser)

        dfq = pd.DataFrame(series)

        outNums = np.transpose(np.percentile(outData, q*100., axis=1))
        for i in range(outData.shape[-1]):
            dfq[f'out_{i}'] = outNums[i]

        dfq.name = q
        values.append(dfq)
        printmd(f'\nIn + Out data **{q:.2f} quantile**')
        print(dfq)

#==========================================================================
def CalcFavScores(config, prices):
    """
    prices should have shape (stocks, timesteps)
    out will have shape (stocks, timesteps, numPeriods)
    The number of periods is defined by the length of config['outputRanges']
    Can use this to output short, medium and long term scores
    """
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
    Translate a vector to the left (neg) or right (pos)
    Padding is a single value, that the vector will be padded with
    Set padding==-2 to use the closest value
    Output has same length as input (some padding values added, some removed)
    Returns (output, valid)
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
def ScaleData(dfs, cols, quantile=0.90):
    """
    dfs is a list of data frames
    cols is a list of the column names that need to be scaled
    All values in all dataframes are scaled by the same ratio
    quantile is 0. to 1. This quantile will scaled to =1
    """
    if not isinstance(cols, list):
        cols = [cols]
    
    # Determine the average of the 90th quantiles
    vals = []
    for df in dfs:
        for col in cols:
            vals.append(df[col].abs().quantile(quantile))
    
    # Get the scaler
    scaler = 1 / np.mean(vals)
    
    # Apply the scaler
    for df in dfs:
        for col in cols:
            df.loc[:, col] *= scaler

#==========================================================================
def ShiftAndScaleCol(col):
    """
    col is a pandas column name to adjust
    Returns a shifted and scaled column
    Usage:
    df.loc[:, 'scaled'] = ShiftAndScaleCol(df.loc[:, 'close'])
    Note that with this method, the column of the dataframe is scaled
    independently from all other data, such that it has
    a mean of 0 and scaled such that 90th percentile = 1
    """
    temp = col - col.mean()
    return temp * (1 / temp.abs().quantile(0.90))

#==========================================================================
def AddVix(r, dfs, prices):
    """
    ## VIX
    Add volatility Index to a list of data frames
    """
    volatility = CalcVolatility(r.config, prices)
    for i in np.arange(r.sampleCount):
        for j in np.arange(volatility.shape[-1]):
            col = 'vix{}'.format(j+1) # Column name
            dfs[i].loc[:, col] = volatility[i,:,j]
    
    # Volatility Normaliser
    # Scale all by the same amount
    vixCols = [col for col in dfs[0].columns if 'vix' in col]
    ScaleData(dfs, vixCols)

#==========================================================================
def MakeExpSpacedPeriods(numPeriods, maxDaysPast, firstPeriodLen, nonlinearity=3):
    """Makes a set of non-overlapping periods over some index range. The lengths
    of the periods will be increasing exponentially.

    Args:
        numPeriods (int): [description]
        maxDaysPast (int): [description]
        firstPeriodLen (int): [description]
        nonlinearity (int, optional): [description]. Defaults to 3.

    Returns:
        [type]: [description]
    """
    seeds = np.linspace(0,1,numPeriods)
    b = nonlinearity # in y = a*exp(b*x) + c. Affects linearity. 4 is good. Lower: more linear
    a = (maxDaysPast-firstPeriodLen) / (np.exp(b)-1)
    c = firstPeriodLen-a
    periodStartsB4Today = np.round(a*np.exp(seeds*b)+c).astype(int)
    periodLengths = np.concatenate(([firstPeriodLen], np.diff(periodStartsB4Today))).astype(int)       
    return periodStartsB4Today, periodLengths
            
#==========================================================================
def CalcVolatility(config, prices):
    """
    ## Volatility (erraticism)
    Use an integral of the differential squared to measure this
    Measure it over a large period, and also measure the comparative
    erraticism of different time periods
    """
    
    # Absolute Erraticism Score
    samples = prices.shape[-2]
    timesteps = prices.shape[-1]
    
    # Comparative Erraticism
    # Set up the time periods
    numPeriods = config['vixNumPastRanges']
    maxDaysPast = config['vixMaxPeriodPast']
    firstPeriodLen = 5

    volatility = np.zeros((samples, timesteps, numPeriods))
    if numPeriods == 0:
        return volatility

    # Legacy periods (as of 2022-01-10)
    # periodStartsB4Today, periodLengths = MakeExpSpacedPeriods(numPeriods, maxDaysPast, firstPeriodLen)
    # # Make all periods extend up to the present time
    # # This means the periods overlap
    # # A field that says 'what was the vix 10-14 hours ago' is probably less useful than a field 'what was the vix 0-14 hours ago'
    # for i in range(numPeriods):
    #     periodLengths[i] = periodStartsB4Today[i]
    
    periodStartsB4Today = np.geomspace(firstPeriodLen, maxDaysPast, numPeriods, dtype=int)
    periodLengths = periodStartsB4Today
    
    # Volatility
    # Diferences in the log2 domain are ratios, so the result is scale-agnostic (no further normalisation needed)
    noise = np.concatenate((np.square(np.diff(np.log2(prices))), [[0]]*samples), axis=1)
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
    ### RSI (relative strength index)
    This is essentially the RSI (exponential method), but represented differently
    Real RSI = 100.0 - (100.0 / (1.0 + RS)). It ranges from 0-100 (human intuitive)
    I calculate a number that hovers around -1 to 1 (machine intuitive).
    dfs is a list of data frames; a data frame for each sample
    """
    def CalcRSI(ser, windowLen):
        # ser is a data series
        # Get the difference in price from previous step
        delta = ser.diff()
        # Get rid of the first row, which is NaN since it did not have a previous 
        # row to calculate the differences
        delta = delta[1:]
        
        # Make separate series for the positive gains (up) and negative gains (down)
        deltaUp, deltaDown = delta.copy(), delta.copy()
        deltaUp[delta < 0] = 0
        deltaDown[delta > 0] = 0
        
        # Exponential weighted moving average
        # Calculate the EWMA
        # 2 lines below are deprecated
#        roll_up1 = pandas.stats.moments.ewma(up, span=windowLen)
#        roll_down1 = pandas.stats.moments.ewma(down.abs(), span=windowLen)
        roll_up1 = deltaUp.ewm(span=windowLen,min_periods=0,adjust=True,ignore_na=True).mean()
        roll_down1 = deltaDown.abs().ewm(span=windowLen,min_periods=0,adjust=True,ignore_na=True).mean()
        
        # Calculate the RSI based on EWMA
        RSI = roll_up1 / roll_down1
        return np.log(RSI)
    
    newCols = []
    for df in dfs:
        for i, windowLen in enumerate(r.config['rsiWindowLens']):
            col = f"rsi{i}_{windowLen}"
            newCols.append(col)
            df.loc[:, col] = CalcRSI(df['close'], windowLen)
            df.loc[:, col] = df[col].replace([np.inf, -np.inf, np.nan], [2, -2, 0])
    
    ScaleData(dfs, newCols)
    
    return

#==========================================================================
def AddEma(r, dfs):
    """
    ## Exponential Moving Average relative to price
    Add Exponential Moving Averages to the data frames
    """
    newCols = []
    for df in dfs:
        for i, span in enumerate(r.config['emaLengths']):
            col = f'ema{i}_{span}'
            newCols.append(col)
            df.loc[:,col] = df['close'].ewm(span=span,min_periods=0,adjust=True,ignore_na=False).mean() / df['close'] - 1

    ScaleData(dfs, newCols)
    return

#==========================================================================
def AddDivergence(r, dfs):
    """
    ## Adds Divergence columns to the data frames
    I define divergence as the price relative to the moving average of X points
    """
    newCols = []
    for df in dfs:
        for i, span in enumerate(r.config['dvgLengths']):
            col = f'dvg{i}_{span}'
            newCols.append(col)
            df.loc[:,col] = df['close'] / df['close'].rolling(window=span, min_periods=0).mean() - 1

    for col in newCols:
        ScaleData(dfs, col)
    return

#==========================================================================
def AddSpread(r, dfs):
    """
    ## Adds Spread column to each df
    Spread = (high - low) / close
    """
    for df in dfs:
        df.loc[:, 'spread'] = (df.loc[:, 'high'] - df.loc[:, 'low']) / df.loc[:, 'close']

    ScaleData(dfs, ['spread'])
    return

#==========================================================================
def AddLogDiff(r, dfs):
    """
    ## Diff Log Series
    Add a logarithmic differential series to the data frames
    """

    for df in dfs:
        col2 = np.log2(np.array(df['close']))
        df.loc[:,'logDiff'] = np.diff(col2, prepend=col2[0])

    ScaleData(dfs, ['logDiff'])
    
    return

#==============================================================================
def ScaleVolume(dfs):
    # Normalise the volume
    for df in dfs:
        df.volume /= df.volume.quantile(0.9)

#==============================================================================
def PrepHighLowData(dfs):
    """
    dfs must be a list of DataFrames
    Prepares high and low columns for entry into the neural network
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
    
    # Normalisation high and low equally over all stocks
    ScaleData(dfs, ['high', 'low'])
           
    return

