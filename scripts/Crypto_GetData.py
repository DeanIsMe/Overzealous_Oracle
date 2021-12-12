# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:32:54 2018

@author: Dean
"""
#%%
import numpy as np

import pandas as pd
import time

import pycwatch # https://github.com/iuvbio/pycwatch
# API docs are here: https://docs.cryptowat.ch/rest-api/markets/list
import pandas as pd

from private_keys import cryptowatch_public

from IPython.display import Markdown, display
def printmd(string, color=None):
    if color is None:
        display(Markdown(string))
    else:
        colorstr = "<span style='color:{}'>{}</span>".format(color, string)
        display(Markdown(colorstr))


# %%
#==========================================================================
# Get hourly dataframe
# Returns a dataframe for each coin. Attempts to download numHours of data
# for each coin. If that much data isn't available, then it will return
# what it finds
# close, high, low, volume
def GetHourlyDf(coins, numHours):
    """
    Returns a list of dataframes
    Columns in output are close, high, low
    """
    t_diff_max = 40*24*60*60 # Seconds. Max of 1000 points, so I'll do only 40 days per call (960 hours)
    dfs = []
    t_now = int(time.time())


    api = pycwatch.rest # create api client
    api.api_key = cryptowatch_public

    # list of available assets
    #assets = api.list_assets()
    #price = api.get_market_price(exchange, pair)

    # Get all market info:
    #all_markets = api.get_all_market_summaries()
    # all_markets is a dictionary with ~7828 entries
    # all_markets['binance:cakebtc'] =
    #     {'price': {'last': 0.0006343,
    #     'high': 0.0006895,
    #     'low': 0.0006132,
    #     'change': {'percentage': -0.0031431714600031, 'absolute': -2e-06}},
    #     'volume': 916706.98,
    #     'volumeQuote': 590.520499377}

    # Generate a list of every exchange, and the number of pairs on each
    # This will be used to determine precedence
    all_pairs = api.list_markets()
    all_pairs_df = pd.DataFrame(all_pairs)
    exch = all_pairs_df['exchange'].drop_duplicates()
    pair_count = [sum(all_pairs_df['exchange']==e) for e in exch]
    exch_precedence = dict(zip(list(exch),pair_count)) # exchanges with higher precedence will be use preferably

    period_options = {\
        '1m':60, 
        '3m':180,
        '15m':900,
        '30m':1800,
        '1h':3600,
        '2h':7200,
        '4h':14400,
        '6h':21600,
        '12h':43200,
        '1d':86400
        # There are further opitions, but I won't use them
    }
    ohlc_col_headers = [
    'time_unix',
    'open', # price
    'high', # price
    'low', # price
    'close', # price
    'volume', # volume, in units of the base coin
    'quote_volume' # volume, in units of the quoted coin
    ]

    period_str = '1h'
    period_int = period_options[period_str]

    def FindDataFirstFinalTimes(exchange, pair_str, period_int):
        # Find the first and final times available for this data pair
        t_2010 = 1262332800 # Unix time for 2010
        t_now = int(time.time())
        # Get the first data point
        try:
            ohlc_resp = api.get_market_ohlc(exchange, pair_str, periods=period_int, after=t_2010)
        except pycwatch.errors.APIError:
            print(f'[{coin}] Exchange {exchange} with pair {pair_str} not found! Something went wrong')
            raise
        if len(ohlc_resp[period_str]) == 0:
            return (t_now, t_now)
        this_df = pd.DataFrame(ohlc_resp[period_str], columns = ohlc_col_headers)
        first_t = this_df['time_unix'].iloc[0]
        # Get the final data point
        ohlc_resp = api.get_market_ohlc(exchange, pair_str, periods=period_int, before=t_now)
        this_df = pd.DataFrame(ohlc_resp[period_str], columns = ohlc_col_headers)
        final_t = this_df['time_unix'].iloc[-1]
        return (first_t, final_t)

    print('\n********************************************************')
    printmd(f"## Get Crypto Data")
    print(f"Getting {numHours} hours ({numHours/24} days)")
    print(f"Starting from {time.ctime(t_now - numHours * 60 * 60)} until now")

    for coin in coins:
        printmd(f'### [{coin}] Getting data for {coin}')

        asset_details = api.get_asset_details(coin) # Lists the details of every trading pair that the coin is involved with
        pairs_df = pd.DataFrame(asset_details['markets']['base']) # All trading pairs where this coin is "base" currency
        pairs_df['exch_score'] = pairs_df['exchange'].map(lambda e : exch_precedence[e] if e in exch_precedence else 0)
        pairs_df.sort_values('exch_score', ascending=False)

        # Decide on the exchange and coin pair
        base_options = ['usdt', 'usd'] # Could expand on this to support BTC and ETH pair
        option_count = 0

        # Check the length of data available for every possible exchange & pair
        pairs_df['first_t'] = int(0)
        pairs_df['final_t'] = int(0)
        pairs_df = pairs_df.astype({'exchange':str, 'exch_score':int, 'pair':str,'first_t':'int64', 'final_t':'int64'})
        for base in base_options:
            pair_str = coin.lower() + base
            for pair_idx in range(len(pairs_df)):
                if pairs_df['pair'][pair_idx] == pair_str:
                    # Found the pair. How long has it existed for?
                    (first_t, final_t) = FindDataFirstFinalTimes(pairs_df['exchange'][pair_idx], pair_str, period_int)
                    pairs_df.at[pair_idx, 'first_t'] = first_t
                    pairs_df.at[pair_idx, 'final_t'] = final_t
                    option_count += 1
                    #print(f"{pair_str:8} from {exchange:12} avail for {(final_t - first_t) / (3600*24):6.0f} days. Since {time.ctime(first_t)}")
        if option_count == 0:
            print(f'ERROR! No suitable pair found for coin {coin}')
            continue

        # Choose from the options
        pairs_df['days'] = (pairs_df['final_t'] - pairs_df['first_t']) / (60*60*24)
        t_range = [int(t_now - numHours * 3600), t_now] # The requested time range
        pairs_df['time_coverage'] = (pairs_df['final_t'].map(lambda t : max(t_range[0], min(t_range[1], t))) - \
            pairs_df['first_t'].map(lambda t : min(t_range[1], max(t_range[0], t)))\
             + pairs_df['exch_score']) / (t_range[1] - t_range[0])
        # I've added 'exch score' to the time coverage such that if 2 exchanges have the same
        # time coverage, the one with the higher score will be chosen
        # Print a table of all of the candidate pairs & exchanges
        print(pairs_df[pairs_df['first_t'] != 0][['exchange','pair','days','time_coverage']].to_string(
            formatters={'time_coverage':'{:6.3f}'.format, 'days':'{:.2f}'.format}))

        # Choose the best option
        pair_idx = pairs_df['time_coverage'].idxmax()
        exchange = pairs_df['exchange'][pair_idx]
        pair_str = pairs_df['pair'][pair_idx]
        first_t = pairs_df['first_t'][pair_idx]
        final_t = pairs_df['final_t'][pair_idx]
        
        # Print chosen pair & exchange
        hours_avail = int((final_t - first_t) / 3600)
        print(f'[{coin}] Using pair {pair_str} from exchange {exchange}. {hours_avail/24:.0f} days of data')
        if hours_avail < numHours:
            printmd(f'[{coin}] **{numHours/24:.0f} days requested, but only {hours_avail/24:.0f} days available!**', color="0xFF8888")
        
        # Collect the actual data
        # Start time at the back and work forward
        # unix time (second since 1970)
        t = first_t
        print(f"[{coin}] Starting from {time.ctime(t)}")
        df = pd.DataFrame()
        conn_err_count = 0 # count of consecutive connection errors
        while t < final_t:
            t_start = t
            time_span = min(t_diff_max, final_t - t)
            point_count = int(time_span/3600)
            t_end = t + time_span
            
            print(f'[{coin}] Getting {point_count:4.0f} points, up to time {time.ctime(t_end)}')

            try:
                ohlc_resp = api.get_market_ohlc(exchange, pair_str, periods=period_int, after=int(t_start), before=int(t_end))
            except pycwatch.errors.APIError:
                print(f'[{coin}] Exchange {exchange} with pair {pair_str} not found! Something went wrong')
                success = False
                raise
            except: # [ConnectionError, ConnectionAbortedError]:
                print(f'[{coin}] Request data error! Trying again')
                conn_err_count += 1
                if conn_err_count > 3:
                    print(f'[{coin}] Giving up')
                    success = False
                    break
                continue

            conn_err_count = 0
                

            points_recv = len(ohlc_resp[period_str])
            if points_recv == 0:
                print(f'[{coin}] RECEIVED NO DATA POINTS!')

            if points_recv > 0:
                this_df = pd.DataFrame(ohlc_resp[period_str], columns = ohlc_col_headers)
                this_df['time'] = pd.to_datetime(this_df['time_unix'],unit='s')
                this_df.index = this_df['time']
                this_df.pop('time_unix')
                this_df.pop('time')
                this_df.pop('quote_volume')
                this_df.pop('open')
                
                if points_recv != point_count:
                    print(f'[{coin}] Actually received {points_recv} / {point_count} points.')
            
                if df.empty:
                    df = this_df
                else:
                    df = pd.concat([df, this_df])
                success = True

            t = t_end

        
        if success:
            dfs.append(df)
            print(f'[{coin}] download finished. {len(df)} rows (hours) total')

    # TODO Prune all arrays to have the same max length !@#
    return dfs


# Testing
if __name__ == '__main__':
    dfs = GetHourlyDf(['ETH','BTC'], 2500)

