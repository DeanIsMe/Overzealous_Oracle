# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:32:54 2018

@author: Dean
"""
#%%
import numpy as np

import pandas as pd
import time
import pickle

from private_keys import cryptowatch_public

from DataTypes import printmd

#*******************************************************************************
class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        filehandler = open(filename, 'rb')
        self.package = pickle.load(filehandler)
        filehandler.close()

    #*******************************************************************************
    def GetDurationAvailable(self, pair):
        dur_avail = data[pair]['time'].iloc[-1] - data[pair]['time'].iloc[0]
        return dur_avail

    #*******************************************************************************
    def GetPairsSummary(self):
        data = self.package['data']
        pairs = []
        dur_avail = []
        last_time = []
        for pair in data:
            pairs.append(pair)
            dur_avail.append(data[pair]['time'].iloc[-1] - data[pair]['time'].iloc[0])
            last_time.append(data[pair]['time'].iloc[-1]) 

        df = pd.DataFrame({'pair':pairs, 'dur_avail':dur_avail, 'last_time':last_time})
        return df


    #*******************************************************************************
    def GetHourlyDf(self, coins, num_hours, allow_partial=False, verbose=1):
        """Grab the hourly data from file, as a list of DataFrames

        Args:
            coins (list): a list of coins of interest. The reference coin will be automatically chosen (USD)
            num_hours (int): the number of hours (data points) for each coin
            allow_partial (bool): Support returning series with different lengths. False=Require all series to have num_hours

        Returns:
            list: A list of DataFrames. 1 per coin
        """
        data = self.package['data']
        # 'data' is a dictionary where the key indicates the trading pair
        dfs = [] # Output is a list of dataframes

        # For each coin, pick a pair and extract the desired time
        ref_options = ['usd'] # ['usd', 'usdt', 'usdc'] # Order of preference (only allowing 'usd')
        for coin in coins:
            coin_found = False
            max_rows_avail = 0
            for base in ref_options:
                pair_check = coin.lower() + base
                if pair_check in data:
                    # This trading pair exists. Check the duration
                    dur_avail = data[pair_check]['time'].iloc[-1] - data[pair_check]['time'].iloc[0]
                    rows_avail = len(data[pair_check])
                    max_rows_avail = max(max_rows_avail, rows_avail)
                    if rows_avail > num_hours or allow_partial:
                        # Sufficient duration. Go with it!
                        rows_to_get = min(rows_avail, num_hours) # count
                        if verbose > 0:
                            print(f'[{coin}] Using trading pair {pair_check} for {rows_to_get} steps')
                        # Extract the relevant rows and save
                        # copy() to avoid SettingWithCopyWarning error
                        this_df = data[pair_check].iloc[-rows_to_get:].copy()
                        this_df.name = coin
                        dfs.append(this_df)
                        coin_found = True
                        break
            if not coin_found:
                printmd(f'**ERROR!**', color="0xFF8888")
                print(f'GetHourlyDf: No valid pair found for {coin} in file {self.filename}.')
                print(f"max_rows_avail={max_rows_avail}. Wanted {num_hours} rows.")
                raise
        return dfs



    #*******************************************************************************
    def GetHourlyDfPairs(self, pairs, num_hours):
        """Grab the hourly data from file, as a list of DataFrames

        Args:
            pairs (list): a list of pairs to load
            num_hours (int): the number of hours (data points) for each coin

        Returns:
            list: A list of DataFrames. 1 per coin
        """
        data = self.package['data']
        # 'data' is a dictionary where the key indicates the trading pair
        dfs = [] # Output is a list of dataframes

        # For each coin, pick a pair and extract the desired time
        for pair in pairs:
            if not pair in data:
                printmd(f'\n **ERROR!**', color="0xFF8888")
                print(f'GetHourlyDfPairs: Pair {pair} not found in file {self.filename}.')
            # Check the duration
            dur_avail = data[pair]['time'].iloc[-1] - data[pair]['time'].iloc[0]
            rows_avail = len(data[pair])
            if rows_avail > num_hours:
                # Sufficient duration. Go with it!
                # Extract the relevant rows and save
                # copy() to avoid SettingWithCopyWarning error
                this_df = data[pair].iloc[-num_hours:].copy()
                this_df.name = pair
                dfs.append(this_df)
            else:
                printmd(f'\n **ERROR!**', color="0xFF8888")
                print(f'GetHourlyDfPairs: Insufficient time available for pair {pair} in file {self.filename}.')
                print(f"rows_avail={rows_avail}. Wanted {num_hours} rows.")
        return dfs

# %%
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
# pycwatch
# For incremental data. Not used as of 2022-08-22
import pycwatch # https://github.com/iuvbio/pycwatch
# API docs are here: https://docs.cryptowat.ch/rest-api/markets/list

def GetHourlyDfCryptowatch(coins, numHours):
    """
    Get hourly dataframe from Cryptowatch API (using pycwatch lib)
    Returns a dataframe for each coin. Attempts to download numHours of data
    for each coin. If that much data isn't available, then it will return
    what it finds
    Columns in output are close, high, low, volume
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
    'time', # Unix time (secs since 1970)
    'open', # price
    'high', # price
    'low', # price
    'close', # price
    'volume', # volume, in units of the base coin. Base = nominated coin. E.g. NANO in NANO-USD
    'quote_volume' # volume, in units of the quoted coin. Quoted = reference coin. E.g. USD in NANO-USD
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
        first_t = this_df['time'].iloc[0]
        # Get the final data point
        ohlc_resp = api.get_market_ohlc(exchange, pair_str, periods=period_int, before=t_now)
        this_df = pd.DataFrame(ohlc_resp[period_str], columns = ohlc_col_headers)
        final_t = this_df['time'].iloc[-1]
        return (first_t, final_t)

    print('\n********************************************************')
    printmd(f"## Get Crypto Data")
    print(f"Getting {numHours} hours ({numHours/24:.2f} days)")
    print(f"Starting from {time.ctime(t_now - numHours * 60 * 60)} until now")

    for coin in coins:
        printmd(f'### [{coin}] Getting data for {coin}')

        # STEP 1: DECIDE ON THE PAIR & EXCHANGE
        # The output of Step 1 is exchange, pair_Str, first_t, & final_t
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
        printmd(f'[{coin}] **Exchange & pair comparison:**')
        print(pairs_df[pairs_df['first_t'] != 0][['exchange','pair','days','time_coverage']].to_string(
            formatters={'time_coverage':'{:6.3f}'.format, 'days':'{:7.2f}'.format}))

        # Choose the best option
        pair_idx = pairs_df['time_coverage'].idxmax()
        exchange = pairs_df['exchange'][pair_idx]
        pair_str = pairs_df['pair'][pair_idx]
        first_t = pairs_df['first_t'][pair_idx]
        final_t = pairs_df['final_t'][pair_idx]

       
        # Print chosen pair & exchange
        hours_avail = int((final_t - first_t) / 3600) + 1
        printmd(f'[{coin}] Using pair **{pair_str}** from exchange **{exchange}**. {hours_avail/24:.0f} days of data available')
        if hours_avail < numHours:
            printmd(f'[{coin}] **{numHours/24:.0f} days requested, but only {hours_avail/24:.0f} days available!**', color="0xFF8888")
        

        # STEP 2: DOWNLOAD THE DATA
        # Collect the actual data
        # Start time at the back and work forward
        # unix time (second since 1970)
        t = max(first_t, int(t_now - numHours * 3600)) 
        printmd(f'\n[{coin}] **Downloading the data**')
        print(f"[{coin}] Starting from {time.ctime(t)}")
        df = pd.DataFrame()
        conn_err_count = 0 # count of consecutive connection errors
        while t < final_t:
            t_start = t
            time_span = min(t_diff_max, final_t - t)
            point_count = round(time_span/3600)
            t_end = t + time_span
            
            print(f'[{coin}] Getting {point_count:4.0f} points, up to time {time.ctime(t_end)}')

            #print(f'[{coin}] Getting {point_count:4.2f} points, {t_start} to {t_end}. Diff= {t_end - t_start}')

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
                this_df.set_index(inplace=True, keys=pd.DatetimeIndex(pd.to_datetime(this_df['time'],unit='s')))
                this_df.index.name='datetime'
                this_df.pop('quote_volume')
                this_df.pop('open')
                this_df.rename(columns={'volume':'volume_nom'}) # Volume in nominated currency
                
                if points_recv != point_count:
                    print(f'[{coin}] Actually received {points_recv} / {point_count} points.')
                    #print(f'[{coin}] Actually received {points_recv} / {point_count} points. Up to {time.ctime( this_df['time'].iloc[-1] )}') !@#
            
                if df.empty:
                    df = this_df
                else:
                    df = pd.concat([df, this_df])
                success = True

            t = t_end

        if success:
            dfs.append(df)
            print(f'[{coin}] download finished. {len(df)} rows (hours) total')

    # # To save a data set:
    # dateStr = datetime.now().strftime('%Y-%m-%d')
    # filehandler = open(f'./indata/dfs_{len(dfs)}coins_{numHours}hours_{dateStr}.pickle', 'wb')
    # package = {'dfs':dfs, 'coinList':r.coinList, 'numHours':numHours, 'dateStr':dateStr}
    # pickle.dump(package, filehandler)
    # filehandler.close()
    return dfs


#%%
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
# READ & PICKLE KRAKEN CSV
# Downloaded from https://support.kraken.com/hc/en-us/articles/360047543791-Downloadable-historical-market-data-time-and-sales-


# Read in all hourly data

import pandas as pd
import numpy as np
import os
import time
from DataTypes import printmd
import pickle

def ReadKrakenCsv(csv_dir):
    """
    READ & PICKLE KRAKEN CSV
    First, download the full data dump of kraken data from:
    https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data
    Then, call this function to read the hourly CSVs into a dataframe, & pickle it.
    All non-hourly files are currently ignored.
    The intention is that the data file could then be updated as needed from cryptowatch.
    This function also scrubs the data
    """
    printmd('## Read Kraken CSV')

    print(f'Loading CSVs from {csv_dir}')

    #csv_dir = 'C:/Users/deanr/Desktop/temp/kraken_data/Kraken_OHLCVT'
    data_hist = {}

    timestep = int(3600)
    time_modified = None
    time_last_data = None

    for file in os.listdir(os.fsencode(csv_dir)):
        filename = os.fsdecode(file)
        if filename.endswith("_60.csv"):
            pair_str = filename.split('_')[0].lower()
            # Kraken API uses 'XBT' instead of 'BTC'. Apply this change
            pair_str = pair_str.replace('xbt','btc')

            pair_df = pd.read_csv(os.path.join(csv_dir, filename), header=None, \
                names=['time','open','high','low','close','volume_nom','trades'], \
                    dtype={'time':np.int64, 'trades':np.int64})
            # volume_nom = volume in nominated currency. E.g. 'NANO' in 'NANO-USD' pair
            pair_df.set_index(inplace=True, keys=pd.DatetimeIndex(pd.to_datetime(pair_df['time'], unit='s')))
            pair_df.index.name = 'datetime'
            pair_df.pop('open')
            pair_df.pop('trades')
            pair_df['filler'] = False # Indicates whether each row was generated to fill a gap
            data_hist[pair_str] = pair_df
            # print(os.path.join(directory, filename))
            if time_modified is None:
                time_modified = os.stat(os.path.join(csv_dir, filename)).st_mtime

            if pair_str == 'ethusd':
                time_last_data = pair_df['time'].iloc[-1]
            continue
        else:
            continue
    
    # Remove spotty starts and fill gaps in the data
    data_hist, _, _ = DataScrubbing(data_hist, timestep)

    # Add change_vs_market
    total_market_change = CalcChangeVsMarket(data_hist)

    # Save the data
    date_str = pd.to_datetime(time_last_data, unit='s').strftime('%Y-%m-%d')
    save_filename = f'./indata/{date_str}_price_data_60m.pickle'
    filehandler = open(save_filename, 'wb')
    package = {'data':data_hist, 'total_market_change':total_market_change, 'date_str':date_str, \
        'time_saved':time.time(), 'time_last_data':time_last_data, 'time_kraken_modified':time_modified, \
            'gaps_filled_until':time_last_data, 'timestep':timestep}
    pickle.dump(package, filehandler)
    filehandler.close()
    print(f'Saved to {save_filename}')
    print(f'DONE! Loaded & pickled CSV data for {len(data_hist)} pairs.')

#*******************************************************************************
# DEALING WITH GAPS IN DATA
from datetime import datetime

def RemoveSpottyStart(df, timestep, check_len = 24):
    """Removes initial data from a series, until there is a gapless block of data
    of length [check_len].

    Args:
        df ([DataFrame]): The dataframe to modify
        timestep ([int]): Expected timestep in seconds, e.g. 3600 for hour
        check_len (int, optional): The number of required contiguous, gapless rows required to be considered 'valid'. Defaults to 24.

    Returns:
        [tuple]: (df, rows_removed). Returns an empty dataframe if all data is bad
    """
    rows_initial = len(df.index)
    t_ser = df[df['filler'] == False]['time']
    delta_t = np.diff(t_ser)

    # TODO check if I actually want convolution here, or autocorrelation
    gap_sum = np.convolve(delta_t / timestep, np.ones([check_len,]), mode='valid')
    first_good_idx = np.argmin(gap_sum)
    if gap_sum[first_good_idx] < check_len*0.99:
        printmd('# SOMETHING UNEXPECTED HAPPENED!')
        print('Data gaps are smaller than expected. Are there time duplicates?')
    elif gap_sum[first_good_idx] > check_len*1.01:
        # There are no sections of data that pass the criteria
        df = df[0:0]
        return df, rows_initial
    first_good_t = t_ser.iloc[first_good_idx]

    df = df[df['time'] >= first_good_t]
    rows_final = len(df.index)
    rows_removed = rows_initial - rows_final

    return df, rows_removed


#*******************************************************************************
def FillDataGaps(df, timestep):
    """Price data often has gaps in time. I want data to exist at all time points.
    This function finds where gaps exist, then inserts new rows.
    Price data filled in via interpolation.

    Args:
        df ([DataFrame]): DataFrame of stock data to process
        timestep ([int]): Expected seconds between data points. e.g. 3600 for 1hr

    Returns:
        [tuple]: (df, total_gaps_filled)
    """
    t_ser = df['time']

    t = int(t_ser.iloc[0])
    t -= t%timestep

    t_end = t_ser.iloc[-1]

    # Step through every expected time & record those that are missing
    new_times = []
    while t < t_end:
        data_found = t in t_ser.values
        if not data_found:
            new_times.append(t)
        t += timestep
    
    if len(new_times) > 0:
        newdf = pd.DataFrame(new_times, columns=['time'])
        newdf.set_index(inplace=True, keys=pd.DatetimeIndex(pd.to_datetime(newdf['time'], unit='s')))
        newdf.index.name='datetime'
        # Assume volume is 0 during these gaps.
        # All prices will be interpolated
        newdf['volume_nom'] = 0
        newdf['filler'] = True # All of these rows are filler rows

        df = pd.concat([df, newdf])
        df.sort_index(inplace=True)
        df.interpolate(inplace=True) # Fill in price data with interpolations

    return df, len(new_times)

#*******************************************************************************
def DataScrubbing(data, timestep):
    """Calls RemoveSpottyStart and FillDataGaps on each DataFrame

    Args:
        data ([list]): List of dataframes of stock data
        timestep ([int]): Expected seconds between data points. e.g. 3600 for 1hr

    Returns:
        [tuple]: (data, total_gaps_filled)
    """
    printmd("## Data scrubbing")
    timestep = int(timestep)
    printmd(f'timestep={timestep}. {len(data)} pairs')
    total_gaps_filled = 0
    total_rows_removed = 0
    pairs_to_delete = []
    for pair, df in data.items():
        rows_initial = len(df.index)
        if rows_initial < 100:
            print(f"[{pair:9s}] Had {rows_initial:6} rows. Not enough! REMOVED PAIR")
            pairs_to_delete.append(pair)
            continue

        df, rows_removed = RemoveSpottyStart(df, timestep, check_len=24)
        total_rows_removed += rows_removed
        if len(df.index) == 0:
            # All data is invalid
            print(f'[{pair:9s}] Had {rows_initial:6} rows. Too spotty! REMOVED PAIR')
            pairs_to_delete.append(pair)
            continue

        df, rows_added = FillDataGaps(df, timestep)
        if rows_added == 0 and rows_removed == 0:
            print(f'[{pair:9s}] No gaps in data. Leaving as-in.')
        else:
            rows_final = len(df.index)
            data[pair] = df
            total_gaps_filled += rows_added
            print(f'[{pair:9s}] Had {rows_initial:6} rows. Removed initial {rows_removed:6} spotty rows. Added {rows_added:6} rows to fill in gaps. Now {rows_final:6} rows.')

    for pair in pairs_to_delete:
        del data[pair]

    return data, total_rows_removed, total_gaps_filled

#*******************************************************************************
def FileDataScrubbing(filename):
    """Loads data file, calls FillDataGaps, saves file

    Args:
        filename ([str]): The file to load & save
    """
    filehandler = open(filename, 'rb')
    package = pickle.load(filehandler)
    data = package['data']
    filehandler.close()

    data, total_rows_removed, total_gaps_filled = DataScrubbing(data, int(package['timestep']))

    if total_rows_removed == 0 and total_gaps_filled == 0:
        print('No rows removed or gaps filled')
    else:
        package['gaps_filled_until'] = package['time_last_data']
        package['data'] = data
        filehandler = open(filename, 'wb')
        pickle.dump(package, filehandler)
        filehandler.close()

        print(f'Saved to {filename}')

#*******************************************************************************
def PlotGaps(data, pair='ethusd'):
    """Plots gap frequency for information & exploration purposes
    to assess data quality

    Args:
        data ([list]): list of dataframes
        pair (str, optional): Defaults to 'ethusd'.
    """
    import matplotlib.pyplot as plt

    delta_t = np.diff(data[pair]['time'])

    from collections import Counter
    cntr = Counter(delta_t)
    total_cnt = len(delta_t)
    del cntr[3600] # Remove the 1 hour
    if len(cntr) == 0:
        print(f'[{pair}] has no gaps')
        return
    #ax = plt.plot([int(i)/3600 for i in cntr.keys()], [int(i) for i in cntr.values()], 'x')
    fig, ax = plt.subplots()
    ax.plot([int(i)/3600 for i in cntr.keys()], [int(i)/total_cnt * 100 for i in cntr.values()], 'x')
    ax.set(xlabel='Gap [hours]', ylabel='Occurrence [%]', title=f'Time gaps in {pair}')
    #ax.set_yscale('log')
    ax.grid()

# ******************************************************************************
# Comparing performance to market
# Adds the fields 'volume_usd', 'market_volume_fraction', and 'change_vs_market'
# volume_usd is the hourly volume, in USD
# market_volume_fraction is market cap of each token relative to the
# overall crypto market. Also called 'dominance'. Smoothed.
# change_vs_market is how the token performed compared to the overall crypto
# market, on an hourly basis. Each value is a ratio, so call .product() over
# a range to find the total performance difference over that range.
# 
# Also returns market_ratio, which indicates how the overall market performed
# on an hourly basis.
# 
# I can't easily get the market cap of each coin. Can't even easily get the 
# overall crypto market cap. So, instead of weighting coins by their market cap,
# I weight coins by their volume in USD
def CalcChangeVsMarket(data):
    fiat = ['usd', 'eur', 'cad', 'aud', 'gbp', 'chf', 'jpy'] # All fiat currencies

    avg_len = 24 * 30 # hours. Used for smoothing for calculating the 'dominance' of each token

    paired_to_usd = set()
    usd_pairs = set()

    for pair in data.keys():
        nom, ref = GetPairNomRef(pair)
        if ref == 'usd':
            usd_pairs.add(pair)
            paired_to_usd.add(nom)

    # Calculate volume_usd (price * vol) for all USD pairs
    # volume_usd is always in USD
    for pair in usd_pairs:
        df = data[pair]
        df['volume_usd'] = df['close'] * df['volume_nom']
        # print(f"added {pair:8s}  len={len(df['volume_usd'])}")

    # Find pairs that don't include USD, but can be converted to USD
    # This is to make sure that I account for all (most) of the volume
    # This will add pairs using EUR, BTC, USDT, etc.
    for pair, df in data.items():
        nom, ref = GetPairNomRef(pair)
        if ref in paired_to_usd:

            # print(f"Extra pair {pair:8s}")

            df_refusd = data[ref + 'usd'] # Should always succeed, because paired_to_usd is known
            idx_common = df.index.intersection(df_refusd.index)
            df['volume_usd'] = (df['close'] * df['volume_nom']).multiply(df_refusd['close'][idx_common])

            # Add volume_usd to reference currency USD pair
            df_refusd['volume_usd'] = df_refusd['volume_usd'].add(df['volume_usd'], fill_value=0)
            # Add volume_usd to nominated currency USD pair (if it exists)
            if nom + 'usd' in data.keys():
                df_nomusd = data[nom + 'usd']
                df_nomusd['volume_usd'] = df_nomusd['volume_usd'].add(df['volume_usd'], fill_value=0)


    # Smooth and sum to get total volume in USD
    ser_volume_usd = pd.Series(dtype='float64')
    for pair, df in data.items():
        if 'volume_usd' in df.columns:
            # I'm doing ".sum() / avg_len" instead of ".mean()"", because there's different behaviour
            # when fewer than avg_len data points are available. ".sum() / avg_len" will rise from 0.
            df['volume_usd'] = df['volume_usd'].rolling(avg_len, min_periods=1).sum() * 1/avg_len
            nom, ref = GetPairNomRef(pair)
            if (pair in usd_pairs) and (nom not in fiat): # Include only tokens (exclude fiat)
                ser_volume_usd = ser_volume_usd.add(df['volume_usd'], fill_value=0)

    # Scale each volume_usd by the total
    # This will be used as a proxy for % market cap
    for pair in usd_pairs:
        df = data[pair]
        df['market_volume_fraction'] = df['volume_usd'] / ser_volume_usd[df.index]

    # Calculate change_vs_market
    # Calculate a weighted average for the entire market: average the % change in 'close', weighted by the market_volume_fraction.
    # Note that it's not smoothed.
    # The 'market' is the crypto market, denominated in USD. 
    # So comparing pairs that aren't referenced in fiat (e.g. 'nanobtc') vs the 'market' likely isn't insightful.
    # 'change_vs_market' is a series of ratios. To compare 2 dates, perform a product() of all values between them.

    # First, calculate the market performance
    ser_sum = pd.Series(dtype='float64')
    ser_count = pd.Series(dtype='float64')
    for pair in usd_pairs:
        nom, ref = GetPairNomRef(pair)
        if nom not in fiat: # Include only tokens (exclude fiat)
            df = data[pair]
            diff = df['close'].pct_change()
            diff.iloc[0] = 0.
            ser_sum = ser_sum.add(diff * df['market_volume_fraction'], fill_value=0)
            ser_count = ser_count.add(df['market_volume_fraction'], fill_value=0)
    ser_market_diff = ser_sum / ser_count
    ser_market_ratio = ser_market_diff + 1. # Ratios of changes

    # For each coin, for each time step: calculate the performance vs the market
    for pair, df in data.items():
        idx_common = df.index.intersection(ser_market_ratio.index)
        ratio = df['close'].pct_change() + 1
        ratio.iloc[0] = 1.
        df['change_vs_market'] = ratio.divide(ser_market_ratio[idx_common])
    
    print("Calculated market_volume_fraction and change_vs_market")

    # I COULD use unit-less absolute values instead of ratios
    # df['change_vs_market'].cumprod() gives the unit-less absolute values

    # Ratios vs absolute values
    # df['change_vs_market'].cumprod()
    # is the same as
    # df['close'] / ser_market_ratio.cumprod()

    # df['change_vs_market'][20:25].prod()
    # is the same as
    # a[24] / a[20], where:  a = df['close'] / ser_market_ratio.cumprod(); 


    return ser_market_ratio


# ******************************************************************************
# GetPairNomRef 
# Returns the pair string split into (nominated, reference)
# Usage: nom, ref = GetPairNomRef(pair)
# Other terminology: base=nominated.  Quoted=reference.  
# Pair order is nominated-reference. E.g. in NANO-USD, NANO=nominated & USD=reference.
def GetPairNomRef(pair):
    # Logic: all 'reference' pairs are 3 character long, except for usdt and usdc
    if pair[-4:] in ['usdc', 'usdt']:
        return (pair[:-4], pair[-4:])
    else:
        return (pair[:-3], pair[-3:])


#%%
#*******************************************************************************
# Testing
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    print(f'Working directory is "{os.getcwd()}"')

    #filename = './indata/2021-09-30_price_data_60m.pickle'
    #dfs = GetHourlyDf(filename, ['ETH'], 100)

    ReadKrakenCsv('C:/Users/deanr/Desktop/temp/kraken_data/Kraken_OHLCVT 2022 Q2/')


#%%
if 0:
    # Example code to run PlotGaps, or DataScrubbing
    # 2021-12-26
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    filename = './indata/2021-09-30_price_data_60m.pickle'

    filehandler = open(filename, 'rb')
    package = pickle.load(filehandler)
    data = package['data']
    filehandler.close()

    #PlotGaps(data, 'ethusd')
    #data, _, _ = DataScrubbing(data, int(package['timestep']))
# %%
