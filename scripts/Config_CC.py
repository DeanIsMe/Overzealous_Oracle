# -*- coding: utf-8 -*-
# Cryptocurrency configuration
#%%
from DataTypes import FeedLoc, printmd
import numpy as np

#==============================================================================
# GetConfig
# Defines and returns the configuration dict
def GetConfig():
    config = dict()
    

    # ****************************
    # INPUT DATA
    # List of all coins that have a 
    coins_mktcap_001 = ['btc', 'eth', 'eur', 'etc', 'bch', 'xrp', 'xdg', 'ada', 'sol', 'eos', 'ltc', 'link', 'dot', 'xlm', 'xtz', 'zec', 'mana', 'xmr', 'gbp', 'atom', 'matic', 'luna', 'dash', 'algo', 'ape', 'sc', 'fil', 'trx', 'sand', 'flow', 'rep', 'omg', 'ksm', 'oxt', 'avax', 'yfi', 'crv', 'waves', 'lrc', 'uni', 'dai', 'grt', 'gno', 'shib', 'storj', 'aave', 'bat', 'luna2', 'movr', 'enj', 'kava', 'mina', 'nano', 'dydx', 'qtum', 'aud', 'keep', 'axs', 'mln', 'icx', 'comp', 'ant', 'paxg', 'glmr', 'ocean', 'snx', 'knc', 'lpt', 'cqt', 'sushi', 'lsk', 'perp', 'srm', 'sdn', 'chz', 'icp', 'repv2', 'ewt', 'ctsi']
    coins_mktcap_01 = ['btc', 'eth', 'eur', 'etc', 'bch', 'xrp', 'xdg', 'ada', 'sol', 'eos', 'ltc', 'link', 'dot', 'xlm', 'xtz', 'zec', 'mana', 'xmr', 'gbp', 'atom', 'matic', 'luna', 'dash', 'algo', 'ape', 'sc', 'fil', 'trx', 'sand', 'flow', 'rep', 'omg', 'ksm']

    # Raw data
    config['coinList'] = ['BTC','ETH'] # a list of coin strings
    config['numHours'] = 24*365*5

    # ****************************
    # INPUT FEATURES

    # Volatility
    config['vixNumPastRanges'] = 3 # number of ranges to use
    config['vixMaxPeriodPast'] = 14 * 24
    
    # RSI - Relative Strength Index
    config['rsiWindowLens'] = [24, 96] # A list of window lengths for RSI calcs. One input per value in the list
    
    # Exponential Moving Average
    config['emaLengths'] = [] # The span of the EMAs

    # I define divergence as the price relative to the moving average of X points
    config['dvgLengths'] = list(np.geomspace(start=1, stop=180, num=5, dtype=int) * 24)

    config['changeVsMarketLens'] = [1, 24, 24*7, 24*60] # A list of lengths to use in calculating the change vs market


   # ****************************
    # INPUT DATA HANDLING

    config['dataRatios'] = [0.80, 0.20 ,0.00] # Training, Validation, Testing


    # Input data and output data are divided by 90th percentile. Then, they
    # are multiplied by the 'scale'. This has a massive impact on the training
    # and results
    config['inScale'] = 1.

    # feedLoc indicates the location at which the data feature will be fed into the network
    # Note that a feature can be used at multiple feed locations
    flc = [[] for i in range(FeedLoc.LEN)]

    flc[FeedLoc.conv].append('ema')
    flc[FeedLoc.conv].append('dvg')
    flc[FeedLoc.conv].append('volNom')
    flc[FeedLoc.conv].append('logDiff')
    flc[FeedLoc.conv].append('rsi')
    flc[FeedLoc.conv].append('vix')
    flc[FeedLoc.conv].append('vsMarket')
    
    # Add everything everywhere
    # flc[FeedLoc.conv].append('ema')
    # flc[FeedLoc.conv].append('dvg')
    # flc[FeedLoc.conv].append('volNom')
    # flc[FeedLoc.conv].append('logDiff')
    # flc[FeedLoc.conv].append('rsi')
    # flc[FeedLoc.conv].append('vix')

    # flc[FeedLoc.rnn].append('ema')
    # flc[FeedLoc.rnn].append('dvg')
    # flc[FeedLoc.rnn].append('volNom')
    # flc[FeedLoc.rnn].append('logDiff')
    # flc[FeedLoc.rnn].append('rsi')
    # flc[FeedLoc.rnn].append('vix')

    # flc[FeedLoc.dense].append('ema')
    # flc[FeedLoc.dense].append('dvg')
    # flc[FeedLoc.dense].append('volNom')
    # flc[FeedLoc.dense].append('logDiff')
    # flc[FeedLoc.dense].append('rsi')
    # flc[FeedLoc.dense].append('vix')

    flc[FeedLoc.dense].append('market_volume_fraction')


    config['feedLoc'] = flc

    # ****************************
    # OUTPUT (TARGET) DATA

    # Favourability Score: CompareToFutureData
    # config['outputRanges'] = [[1,5], [6,25], [26,125]] # The ranges over which to calculate output scores
    # Starts at 1
    # I'm using hourly data
    config['outputRanges'] = [[1,5], [6,25], [26,125]]
    # Also defines the number of output periods
    
    # How many steps to exclude from training & validating because the 'favourability' score is not well defined
    config['excludeRecentSteps'] = 50 # Tradeoff between how recent, and accuracy to 'favourability' score
    
    # Favourability Score looks X days into the future. When there aren't X days
    # of future data available, the calculation is incomplete. 
    # pullUncertainYTo0 indicates how to handle this incomplete calculation.
    # When pullUncertainYTo0 is true, the favourability score will be pulled
    # more towards zero as the more and more dates are missing. When it's
    # false, the score will be calculated assuming that the price does not vary
    # at all from the last day that data exists
    config['pullUncertainYTo0'] = True

    # Binarise
    # Minimise outliers reduces the size of outliers in output favourability score
    # The binarise =0.1 gives almost no reduction
    # =0.8 gives a huge reduction of outliers
    # =3 makes the data almost binary
    # 0.4 is a good tradeoff for minimising outliers
    config['binarise'] = 0
    
    # Ternarise
    # Transform towards a ternary buy/sell/neutral
    # ternarise = 10 (strict ternary). 
    # = 0.5 is less strict, but approaches binary
    # 1-3 seems preferable
    # Binarise and ternarse must not both be non-zero
    config['ternarise'] = 0

    # selectivity only applies when ternarise is non-zero. Higher=more often neutral
    # selectivity 1=frequent buy/sell signals. 3=very picky buy/sell
    config['selectivity'] = 2
    # A good balance seems to be ternarise = 1, selectivity = 2

    # Input data and output data are divided by 90th percentile. Then, they
    # are multiplied by the 'scale'. This has a massive impact on the training
    # and results
    config['outScale'] = 1.


    # ****************************
    # NEURAL NET (MODEL)

    # SYSTEM 1 : convolution
    config['convType'] = 'filternet' # 'filternet' or 'wavenet' or 'none'
    # FilterNet conv
    # The 3 parameters below can be a list (one value per conv layer), or a scalar (apply to all conv layers)
    # The num of conv layers will be the greatest number of valid layers
    # If any parameter is empty ([] or 0), then there will be no convolutional layers
    config['convDilation'] = [1,2,4,8,16,32,64,128] # Time dilation factors. 
    config['convFilters'] = [80,75,70,65,60,50,40,30] # Number of filters per layer. List or scalar
    config['convKernelSz'] = 10 # Kernel size per filter
    config['convCascade'] = True # True:Series (default). False:Parallel. True:conv modules feed into eachother (with skip connections).  False:conv modules are in parallel - all have same input

    # WaveNet conv
    config['wnStackCount'] = 1 # repeat all module this many times
    config['wnFactor'] = 2 # the kernel_size and dilation factor (usually 2)
    config['wnModuleCount'] = 9 # Each module increases receptive filed by 'factor'. Total receptive field of a stack will be factor**modulecount
    config['wnWidth'] = 48 # Width of all dense and conv layers

    # SYSTEM 2: RNN (LSTM/GRU)
    config['rnnType'] = 'gru' # 'lstm' or 'gru' or 'none'
    config['bottleneckWidth'] = 128 # A dense layer is added before the LSTM/GRU to reduce the size. 0 to disable.
    config['rnnWidths'] = [128] # Number of neurons in each LSTM/GRU layer. They're cascaded. [] to disable

    # SYSTEM 3: Dense/Fully connected
    config['denseWidths'] = [48] # [256, 128, 64, 32, 16] # These layers are added in series after LSTM and before output layers. Default: none

    # Other model parameters
    config['batchNorm'] = True # Batch Normalisation. applies to conv, rnn, and dense layer modules

    config['regularizerType'] = 'l2' # None or 'l1' or 'l2' or 'l1_l2'
    config['regularizationRateL1'] = 0.01 # default 0.01. aka alpha
    config['regularizationRateL2'] = 0.0001 # default 0.01. aka alpha

    config['dropout'] = 0.2 # Dropout_rate of layer applied before each layer. Set to 0 to disable
    

    
    # ****************************
    # TRAINING

    config['epochs'] = 8 # Number of complete passes of the data (subject to early stopping)
    
    config['evaluateBuildStatePoints'] = 500 # The number of timesteps used to build state when predicting values for model validation during training


    # After training, change the weights to the model that had the best 
    # validation score
    config['revertToBest'] = True
    
    config['earlyStopping'] = 0 # 0 = no early stopping. Otherwise, the number
#     of epochs of no improvements before training is stopped (patience)
    
    # Optimizer
    config['optimiser'] = 'adam'

    config['learningRate'] = 0.002 # initial


    # ****************************
    # OTHER

    config['plotWidth'] = 9

    return config


#==============================================================================
# Creates a string that represents the configuration, in markdown. Prints it.
def PrintConfigString(c):
    s = "---  \n"
    s += "## Configuration  \n"
    s += "### Input data  \n"
    s += f"**Coins**: {', '.join(c['coinList'])} for {c['numHours']} hrs ({c['numHours']/24/365:.2f} years)\\"


    s += "### Input features  \n"
    if c['vixNumPastRanges']:
        s += f"**VIX**: {c['vixNumPastRanges']} ranges, up to {c['vixMaxPeriodPast']} hrs  \n"
    
    if c['rsiWindowLens']:
        s += f"**RSI** lengths of "
        for windowLen in c['rsiWindowLens']:
            s += f"{windowLen}, "
        s += "hrs  \n"
    
    if c['emaLengths']:
        s += f"**EMA** lengths of "
        for windowLen in c['emaLengths']:
            s += f"{windowLen}, "
        s += "hrs  \n"

    if c['dvgLengths']:
        s += f"**Divergence** lengths of "
        for windowLen in c['dvgLengths']:
            s += f"{windowLen}, "
        s += "hrs  \n"


    if c['changeVsMarketLens']:
        s += f"**vsMarket** lengths of "
        for thisLen in c['changeVsMarketLens']:
            s += f"{thisLen}, "
        s += "hrs  \n"
    

    s += "### Input handling  \n"
    s += f"Data split training={c['dataRatios'][0]}, validation={c['dataRatios'][1]}, testing={c['dataRatios'][2]}  \n"
    if c['inScale'] != 1.:
        s += f"Output data scale = {c['outScale']}  \n"
    if c['feedLoc'][FeedLoc.conv]:
        s += f"Data fed to conv: {', '.join(c['feedLoc'][FeedLoc.conv])}  \n"
    if c['feedLoc'][FeedLoc.rnn]:
        s += f"Data fed to RNN: {', '.join(c['feedLoc'][FeedLoc.rnn])}  \n"
    if c['feedLoc'][FeedLoc.dense]:
        s += f"Data fed to dense: {', '.join(c['feedLoc'][FeedLoc.dense])}  \n"



    s += "  \n### Output (target) data  \n"
    s += f"**Time ranges**: {c['outputRanges']} hrs. (excl {c['excludeRecentSteps']})  \n"
    if c['binarise']:
        s += f"Binarise={c['binarise']}  \n"
    if c['ternarise']:
        s += f"Ternarise={c['ternarise']}. Selectivity={c['selectivity']}  \n"
    if c['outScale'] != 1.:
        s += f"Output data scale = {c['outScale']}  \n"


    s += "  \n### Neural net (model)  \n"
    if c['convType'].lower() == 'filternet':
        s += f"**FilterNet** convolution, ({'serial' if c['convCascade'] else 'parallel'})  \n"
        s += f"dilation={c['convDilation']}, filterCnt={c['convFilters']}, kernelSz={c['convKernelSz']}  \n"
    elif c['convType'].lower() == 'wavenet':
        s += f"**WaveNet** convolution  \n"
        s += f"  {c['wnModuleCount']} modules of width {c['wnWidth']}, factor={c['wnFactor']}. STACKED {c['wnStackCount']} times.  \n"
    elif c['convType'].lower() == 'none':
        s += "No convolution stage  \n"
    else:
        s += f"# UNKNOWN CONV STAGE! {c['convType']}  \n"

    s += "  \n"
    if c['bottleneckWidth']:
        s += f"Bottleneck to width {c['bottleneckWidth']}  \n"
    
    s += f"RNN type: **{c['rnnType']}**. LayerWidths = {c['rnnWidths']}  \n"
    s += "  \n"
    s += f"**Dense** LayerWidths={c['denseWidths']}  \n"
    s += "  \n"

    s += "**Model properties**  \n"
    if c['batchNorm']:
        s += "BatchNorm ON (all layers)  \n"
    if c['regularizerType'] == 'None':
        s += "No regularizer  \n"
    elif c['regularizerType'].lower() == 'l1':
        s += f"L1 regularizer. Rate={c['regularizationRateL1']}  \n"
    elif c['regularizerType'].lower() == 'l2':
        s += f"L2 regularizer. Rate={c['regularizationRateL2']}  \n"
    elif c['regularizerType'].lower() == 'l1_l2':
        s += f"L1 & L2 regularizer. Rates={c['regularizationRateL1']} & {c['regularizationRateL2']}  \n"

    s+= f"Dropout rate = {c['dropout']}  \n"
    
    s += "  \n### Training process  \n"
    s += f"Train for **{c['epochs']} epochs**"
    if c['earlyStopping']:
        s += f", early stopping with patience {c['earlyStopping']} epochs"
    s += '  \n'
    s += f"Optimiser=**{c['optimiser']}**, learning rate={c['learningRate']}  \n"

    s += "  \n---  \n"
    printmd(s)


if __name__ == '__main__':
    PrintConfigString(GetConfig())
# %%
