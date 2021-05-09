# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Jan 19 23:10:38 2018

@author: Dean
"""

# updated on 2021-05-08
# See kucoin sdk details:
# https://github.com/Kucoin/kucoin-python-sdk

# client.get_all_tickers() is only returning 10 results! It should be returning
# everything.
This script is not updated and will not work. Not worth it with Kucoin's mess


from kucoin.client import Market
import pandas as pd
import numpy as np
import itertools

from private_keys import kucoin_api_key, kucoin_api_secret, kucoin_sandbox_api_key, kucoin_sandbox_api_secret

api_key = kucoin_sandbox_api_key
api_secret = kucoin_sandbox_api_secret

# REAL server (not sandbox)
# print('WARNING! USING REAL KUCOIN SERVER')
#client = Market(url='https://api.kucoin.com')
# client = Market()

# or connect to Sandbox
client = Market(url='https://openapi-sandbox.kucoin.com')
client = Market(is_sandbox=True)

# get symbol kline
klines = client.get_kline('BTC-USDT','1hour')

# Returns a list of lists with shape (1500,7). Each item:
#   [
#       "1545904980",             //Start time of the candle cycle
#       "0.058",                  //opening price
#       "0.049",                  //closing price
#       "0.058",                  //highest price
#       "0.049",                  //lowest price
#       "0.018",                  //Transaction amount
#       "0.000945"                //Transaction volume
#   ],

#%%

markets = ['BTC', 'ETH', 'NEO', 'USDT', 'KCS', 'BCH'] # Currencies to search between
allData = client.get_symbol_list()
symbolList = pd.DataFrame(allData)

# Quote and Base currencies
# 'baseCurrency' is the currency being bought or sold
# 'quoteCurrency is the currency used to represent the baseCurrency's value.
# BASE-QUOTE.  1 BASE = X QUOTE
# E.g. BTC-USDT = 55000. BTC: base. USDT:quote.  1 BTC = 55000 USDT


# Tickers include the offer and bid prices
ticker_resp = client.get_all_tickers()
tickers = pd.DataFrame(ticker_resp['ticker'])
tickers.rename(columns={'buy': 'bid', 'sell': 'offer'}, inplace=True)
tickers['bid'] = tickers['bid'].astype(float)
tickers['offer'] = tickers['offer'].astype(float)

tickers['quoteCurrency'] = tickers['symbol'].map(lambda sym : sym.split('-')[0])
tickers['baseCurrency'] = tickers['symbol'].map(lambda sym : sym.split('-')[1])
df = tickers

marketData = {}
quoteFromBaseCur = {} # List of lists. For each base currency, lists the available quote currencies
for m in markets:
    marketData[m] = df.loc[df['baseCurrency'] == m]
    quoteFromBaseCur[m] = list(marketData[m]['quoteCurrency'])



# %%

for pair in itertools.combinations(markets, r=2):   
    c1 = pair[0] # Coin 1. Base currency
    c2 = pair[1] # Quote currency
    
    coinList = list(set(quoteFromBaseCur[c1]).intersection(quoteFromBaseCur[c2]))
    if len(coinList) == 0:
        continue
    df1 = marketData[c1].loc[marketData[c1]['baseCurrency'].isin(coinList)]
    df2 = marketData[c2].loc[marketData[c2]['baseCurrency'].isin(coinList)]
    
    
    # Sort both coins so that row orders match up
    df1 = df1.sort_values('baseCurrency')
    df2 = df2.sort_values('baseCurrency')
    # Make row indices match up
    df1.index = df1['baseCurrency']
    df2.index = df2['baseCurrency']
    
    # Determine conversions between c1 and c2
    pair_symbol = c1 + '-' + c2
    this_ticker = tickers.loc[tickers['symbol'] == pair_symbol]
    if this_ticker.size != 0:
        c2Bid = this_ticker[:1]['bid'] # Highest bid price
        c2Offer = this_ticker[:1]['offer']
    else:
        pair_symbol = c2 + '-' + c1
        this_ticker = tickers.loc[tickers['symbol'] == pair_symbol]
        if this_ticker.size == 0:
            print(f'Ticker between pair {pair_symbol} not found!', file=sys.stderr)
            continue
        else:
            c2Bid = 1./this_ticker[:1]['offer']
            c2Offer = 1./this_ticker[:1]['bid']
    
    fees = pow(1-0.001, 3)
    # Considering moving c1 -> Coin -> c2 -> c1
    # Sell c1 to buy Coin (buy Coin from lowest offer)
    # Sell Coin to buy c2 (sell Coin to highest bid)
    # Sell c2 to buy c1 (sell c2 to highest bid)
    diffA = (1 / df1['offer'] * df2['bid']) * c2Bid.iloc[0] * fees - 1
    # Sort by max absolute difference first
    diffA = diffA.reindex(diffA.sort_values(ascending=False).index)
    
    
    # Considering moving c2 -> Coin -> c1 -> c2
    # Sell c2 to buy Coin (buy Coin from lowest offer)
    # Sell Coin to buy c1 (sell Coin to highest bid)
    # Sell c1 to buy c2 (buy c2 from lowest offer)
    diffB = (1 / df2['offer'] * df1['bid']) / c2Offer.iloc[0] * fees - 1
    # Sort by max absolute difference first
    diffB = diffB.reindex(diffB.sort_values(ascending=False).index)
    
    # PRINT OUT RESULTS
    print('\n\n**********  {} and {}  **********'.format(c1,c2))
    # Direction A
    numPos = diffA[diffA > 0].count()

    if numPos == 0:
        coin = diffA.index[0]
        print('Direction A: no options. Best gain is {:4.2f}% ({})'.format(diffA[coin]*100, coin))
    else:
        print('\nDIRECTION A. {} -> Coin -> {} -> {}'.format(c1,c2,c1))
        print('{} options'.format(numPos))
        
    for i in np.arange(min(5, numPos)):
        coin = diffA.index[i]
        
        gain = diffA.loc[coin] * 100
        print('\nOPTION {}: {:>5} for {:4.2f}% gain'.format(i+1, coin, gain))
        print(' SELL -->   BUY')
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c1, coin, df1['offer'][coin],coin,c1))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(coin, c2, df2['bid'][coin],coin,c2))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c2,   c1, c2Bid.iloc[0],c2,c1))
        
    # Direction B
    numPos = diffB[diffB > 0].count()

    if numPos == 0:
        coin = diffB.index[0]
        print('Direction B: No options. Best gain is {:4.2f}% ({})'.format(diffB[coin]*100, coin))
    else:
        print('\nDIRECTION B. {} -> Coin -> {} -> {}'.format(c2,c1,c2))
        print('{} options'.format(numPos))
    
    for i in np.arange(min(5, numPos)):
        coin = diffB.index[i]
        
        gain = diffB.loc[coin]*100
        print('\nOPTION {}: {:>5} for {:4.2f}% gain'.format(i+1, coin, gain))
        print(' SELL       BUY')
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c2, coin, df2['offer'][coin],coin,c2))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(coin, c1, df1['bid'][coin],coin,c1))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c1,   c2, c2Offer.iloc[0],c2,c1))
    

print('DONE')
# EXAMPLES FROM LIBRARY

# get currencies
#currencies = client.get_currencies()

# get market depth
#depth = client.get_order_book('KCS-BTC', limit=5)

# get symbol klines
#from_time = 1507479171
#to_time = 1510278278
#klines = client.get_kline_data_tv(
#    'KCS-BTC',
#    Client.RESOLUTION_1MINUTE,
#    from_time,
#    to_time
#)

# place a buy order
# transaction = client.create_buy_order('KCS-BTC', '0.01', '1000')

# get list of active orders
# orders = client.get_active_orders('KCS-BTC')

# get historical kline data from any date range

# fetch 1 minute klines for the last day up until now
#klines = client.get_historical_klines_tv("KCS-BTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

# fetch 30 minute klines for the last month of 2017
#klines = client.get_historical_klines_tv("ETH-BTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

# fetch weekly klines since it listed
#klines = client.get_historical_klines_tv("NEO-BTC", KLINE_INTERVAL_1WEEK, "1 Jan, 2017")
# %%
