import pandas as pd
import numpy as np

def convert_placeholder_text_to_num(text):
    '''
    Parsing helper script. In a lot of investing.com data e.g.6.1 M is used to indicate 6.1 million. This script converts the string 
    into float for machine learning to be carried out. 
    '''
    result = text
    try:
        if text[-1] == 'M':
            result = float(text[:-2]) * 10**6
        elif text[-1] == 'B':
            result = float(text[:-2]) * 10**9
        elif text[-1] == 'K':
            result = float(text[:-2]) * 10**3
        elif text[-1] == '-':
            result = np.nan
        elif text[-1] == 'nan':
            result = np.nan
        elif text[-1] == 'NA':
            result = np.nan
        elif text[-1] == 'x':
            result = float(text[:-1])
        elif text[-1] == '%':
            result = text.replace(",","")
            result = float(result[:-1])*0.01
    except Exception as e:
        # hide outputs
        pass
        #print(e)
       
    return result

def filter_ratios_returns(fratios_df, test_assets):
    '''
    ML_data - financial ratio dataframe, pre-cleaned with convert_placeholder_text_to_num (dataframe)
    test_assets - asset prices of HK stocks (dataframe)
    
    Returns a df that only has data (tickers) that are present in both investing.com financial ratios AND yahoo stock prices
    '''

    ratio_ticker_list = list(fratios_df.Ticker)
    ratio_ticker_list_new = []

    for elem in ratio_ticker_list:
        ticker = f'{elem:04}'+'.HK'
        ratio_ticker_list_new.append(ticker)

    asset_ticker_list = list(test_assets.columns)

    combined_tickers = [value for value in asset_ticker_list if value in ratio_ticker_list_new]

    combined_tickers
    combined_ticker_int = []

    for elem in combined_tickers:
        combined_ticker_int.append(int(elem[:-3]))

    df = fratios_df[fratios_df['Ticker'].isin(combined_ticker_int)]

    return df