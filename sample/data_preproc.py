import pandas as pd
import numpy as np
from sklearn.utils import shuffle

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

def filter_ratios_returns(fratios_df, returns_df):
    '''
    fratios_df - financial ratio dataframe, pre-cleaned with convert_placeholder_text_to_num (dataframe). Normally "ML_data".
    returns_df - asset prices of HK stocks (dataframe). Normally "test_assets"
    
    Returns a df that only has data (tickers) that are present in both investing.com financial ratios AND yahoo stock prices
    '''

    ratio_ticker_list = list(fratios_df.Ticker)
    ratio_ticker_list_new = []

    for elem in ratio_ticker_list:
        ticker = f'{elem:04}'+'.HK'
        ratio_ticker_list_new.append(ticker)

    asset_ticker_list = list(returns_df.columns)

    combined_tickers = [value for value in asset_ticker_list if value in ratio_ticker_list_new]

    combined_ticker_int = []

    for elem in combined_tickers:
        combined_ticker_int.append(int(elem[:-3]))

    df = fratios_df[fratios_df['Ticker'].isin(combined_ticker_int)]

    # final clean up step: 1) remove duplicates 2) apply bfill to remove NaNs, followed by ffill
    df = df.drop_duplicates(subset='Ticker')
    df = df.bfill(axis=1)
    df = df.ffill(axis=1)

    return df

def extract_financial_ratio(fratio, ML_dataframe):
    '''
    Given a financial ratio (list below), return a dataframe [X1,X2,...,y1,y2,...] where Xi are the financial ratios and yi are the % returns for the asset.
    Vertical axis is time (in decreasing order)
    
    Financial ratios: 
    - EV
    - FCF
    - EBITDA
    - Revenue
    - ROE
    - Gross-Profit-Margin
    - Quick-Ratio
    - Debt / Equity

    Note that FQ corresponds to 2022-12-31 and FQ-1 corresponds to the preceding quarter etc. 
    '''
  
    
    df = ML_dataframe.loc[:,ML_dataframe.columns.str.contains('Ticker') | ML_dataframe.columns.str.contains(fratio)]
    df = df.set_index('Ticker')
    df = df.transpose()
    df = df.pct_change(-1)
    
    return df

def get_returns(asset_prices_df):
    '''
    From asset prices dataframe defined above, do the following:
    - reverse its order
    - calculate percent returns
    - restrict period to between pd.Timestamp('2023-03-31'):pd.Timestamp('2020-03-31') # this is the period for which 
    company valuation metrics have been obtained
    '''
    
    df = asset_prices_df[::-1].pct_change(-1)[pd.Timestamp('2023-03-31'):pd.Timestamp('2020-03-31')]
    
    return df

def ticker_to_fratio_frame(ticker, fratio_df, returns_df,shift=-1):
    '''
    ticker - int obtained from cols: extract_financial_ratio('EV',ML_final).columns
    fratio_df - this is ML_final as above
    returns_df - this is usually get_returns(asset_prices)
    
    shift - -1 (default, returns are coincident with company valuation metrices), 
        0 -  (returns lead financial ratios by one time period leading)
        1 - (" 2 time periods etc.)
    '''
    
    col_names = ['EV','FCF','EBITDA','Revenue','ROE','Gross-Profit-Margin','Quick-Ratio','Debt / Equity', 'Returns']

    ticker_returns_df = f'{ticker:004}'+'.HK'

    # get financial ratios
    #for fratio in financial_ratios:
        # create temp dfs
    EV_tmp = extract_financial_ratio('EV',fratio_df)
    FCF_tmp = extract_financial_ratio('FCF',fratio_df)
    EBITDA_tmp = extract_financial_ratio('EBITDA',fratio_df)
    REV_tmp = extract_financial_ratio('Revenue',fratio_df)
    ROE_tmp = extract_financial_ratio('ROE',fratio_df)
    GPM_tmp = extract_financial_ratio('Gross-Profit-Margin',fratio_df)
    QR_tmp = extract_financial_ratio('Quick-Ratio',fratio_df)
    DE_tmp = extract_financial_ratio('Debt / Equity',fratio_df)

    df = pd.concat([EV_tmp[ticker].reset_index(drop=True),FCF_tmp[ticker].reset_index(drop=True),\
          EBITDA_tmp[ticker].reset_index(drop=True), REV_tmp[ticker].reset_index(drop=True),\
              ROE_tmp[ticker].reset_index(drop=True), GPM_tmp[ticker].reset_index(drop=True),\
              QR_tmp[ticker].reset_index(drop=True), DE_tmp[ticker].reset_index(drop=True),\
                   returns_df[ticker_returns_df].shift(shift).reset_index(drop=True)],axis=1)

    df.columns = col_names

        #df = extract_financial_ratio(fratio,ML_final)
    return df



class FRatioMLdata:
    def __init__(self,fratios_df, returns_df,sector=None,returns_lead_by=-1):
        '''
        Creates an object that easily returns training data [X y] for machine learning. Can filter the original 
        dataframes by GICS sector, as well as leading the returns relative to financial ratio.

        For the purpose of this project, only looking at the time forecasting scheme of t_n -> t_k where k>n. 

        fratios_df - dataframe containing financial ratios
         
        returns_df - dataframe containing returns

        sector - filter the input dataframe by a particular GICS sector. Options are:
                ['Information Technology',
                'Consumer Discretionary',
                'Energy',
                'Financials',
                'Industrials',
                'Communication Services',
                'Healthcare',
                'Consumer Staples',
                'Real Estate']
                
                (default - None, so the whole dataset is returned)

        returns_lead_by - the period (quarter) by which to lead the returns relative to the financial ratio.
                        -1 - returns are coincident and don't lead financial ratios
                        0 - returns lead by 1 quarter
                        1 - returns lead by 2 quarters
                        ...
                        3 - returns lead by 4 quarters

                        (default - -1 and the returns coincide with financial ratios by date, )
        '''

        self.sector = sector
        self.returns_shift = returns_lead_by
        self.fratios_df = fratios_df
        self.returns_df = returns_df

    def transform(self):
        '''
        Apply sector filter and shift the returns by specified period. Finally, drop NaNs and return the dataframe.
        '''
        if self.sector is not None:
            self.fratios_df_filtered = self.fratios_df[self.fratios_df['Sector'] == self.sector]
        else:
            self.fratios_df_filtered = self.fratios_df

        tickers = extract_financial_ratio('EV',self.fratios_df_filtered).columns
        df = pd.DataFrame()
        for ticker in tickers:
            df = pd.concat([df,ticker_to_fratio_frame(ticker,self.fratios_df_filtered, get_returns(self.returns_df),self.returns_shift)])
        
        # remove NaNs, infinities
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        self.train = df.dropna()

        return df
        

    def shuffle(self,random_state=0):
        '''
        Apply sklearn shuffle to rows. Default random_state to 0
        '''
        self.train = shuffle(self.train,random_state=random_state)
