# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:01:52 2018

@author: Dean
"""

import numpy as np
import pandas as pd
import crycompare as cc

# BINANCE
btcHas = ['ETH','TRX','NEO','XRP','NEBL','BNB','BCD','VEN','EOS','ARN','ADA','ICX','WTC','LTC','BCC','XLM','ELF','XVG','IOTA','APPC','QSP','WTC','GAS','QTUM','HSR','POE','BTS','ZRX','TRIG','BQX','WABI','XMR','INS','BTG','OMG','FUEL','VIBE','LEND','GTO','SUB','LSK','BRD','STRAT','ZEC','DASH','ENG','TNB','REQ','TNT','CND','POWR','AION','KNC','CTR','ENJ','FUN','LRC','AST','BCPT','NULS','MTH','DGD','SALT','SNT','OST','BAT','LINK','MCO','WINGS','MANA','NAV','CMT','RCN','GVT','EDO','CDT','PPT','GXS','WAVES','MOD','ARK','XZC','DNT','LUN','MTL','RDN','KMD','SNGLS','EVX','YOYO','ICN','VIB','RLC','MDA','SNM','AMB','DLT','STORJ','OAX','ADX','BNT']
ethHas = ['TRX','NEO','XRP','EOS','VEN','ADA','BNB','NEBL','ICX','XLM','IOTA','BCD','LTC','ARN','XVG','BCC','ELF','ZRX','APPC','QSP','WTC','REQ','QTUM','POE','ETC','OMG','LEND','FUEL','WABI','BQX','SUB','BTS','LRC','ENG','XMR','KNC','AION','INS','TNB','LINK','GTO','TRIG','DASH','AST','FUN','POWR','OST','VIBE','ENJ','TNT','CND','ZEC','BTG','DGD','LSK','HSR','STRAT','SNT','MTH','CTR','SALT','BAT','RDN','NULS','BCPT','CMT','ICN','PPT','CDT','BRD','GXS','RCN','ARK','GVT','MOD','SNGLS','MCO','MANA','NAV','AMB','DNT','WAVES','OAX','MDA','EDO','SNM','XZC','KMD','LUN','MTL','BNT','EVX','STORJ','VIB','DLT','ADX','WINGS']
bnbHas = ['ETH','NEO','NEBL','LTC','VEN','XLM','BCC','ICX','APPC','QSP','WABI','TRIG','BTS','CND','WTC','LSK','GTO','NULS','CMT','BCPT','POWR','BAT','RCN','AION','XZC','MCO','NAV','AMB','OST','RDN','BRD','WAVES','ADX','RLC','DLT']

binanceCoins = [btcHas, ethHas, bnbHas]
binanceCoinsIndex = ['BTC','ETH','BNB']


# Select Binance
exchange = 'BINANCE'
allCoins = binanceCoins
allCoinsIndex = binanceCoinsIndex


idx1 = 0
idx2 = 1
c1 = allCoinsIndex[idx1]
c2 = allCoinsIndex[idx2]
coinList = list(set(allCoins[idx1]).intersection(allCoins[idx2]))

# !@#$ DEAL WITH MAX SIZES OF COIN LIST (can only hold 75 3 char coins)
coinList = coinList[-50:]

success = False

while not success:
    allPrices = cc.Price().priceMulti(coinList, [c1,c2], e=exchange, tryConversion=False)

    # Remove Errors
    if 'Response' in allPrices:
        # A coin wasn't found
        msg = allPrices['Message']
        i = msg.find('coin pair (') + 11
        j = msg[i:-1].find('-')
        coinList.remove(msg[i:i+j])
        print('\nRemoving from coinList: ' + msg[i:i+j])
    else:
        success = True
    

df = pd.DataFrame(allPrices)
c2toc1 = cc.Price().price(c2, c1, e=exchange, tryConversion=False)
if 'Response' in c2toc1:
    c1toc2 = cc.Price().price(c1, c2, e=exchange, tryConversion=False)
    c2toc1 = {c1 : 1/c1toc2[c2]}
    
df.loc['toc2'] = df.loc[c1] / c2toc1[c1]
df.loc['diff'] = (df.loc['toc2'] - df.loc[c2]) / df.loc[c2]

diff = pd.Series(df.loc['diff'])
# Sort by max absolute difference first
diff = diff.reindex(diff.abs().sort_values(ascending=False).index)

#Print the top 5
for i in np.arange(5):
    coin = diff.index[i]
    
    if diff.loc[coin] > 0:
        gain = diff.loc[coin]*100
        print('\nOPTION {}: {:>5} for {:4.2f}% gain'.format(i+1, coin, gain))
        print(' SELL       BUY')
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c2, coin, df[coin][c2],coin,c2))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(coin, c1, df[coin][c1],coin,c1))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c1,   c2, c2toc1[c1],c2,c1))
    else:
        gain = (1/(1+diff.loc[coin])-1)*100
        print('\nOPTION {}: {:>5} for {:4.2f}% gain'.format(i+1, coin, gain))
        print(' SELL -->   BUY')
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c1, coin, df[coin][c1],coin,c1))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(coin, c2, df[coin][c2],coin,c2))
        print('{:>5} --> {:>5} for {:>9} {:>5}/{:>5}'.format(c2,   c1, c2toc1[c1],c2,c1))
    
    
    



