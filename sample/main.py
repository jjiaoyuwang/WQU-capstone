import os
import sys
import pandas as pd
import numpy as np

# IMPORT FUNCTIONS

import data_preproc

# LOAD FINANCIAL RATIOS AND ASSET PRICES
test_merge = pd.read_excel('../jupyter-notebooks/test_manual.xlsx')
test_merge = test_merge.loc[:, test_merge.columns != 'Unnamed: 0']
test_assets = pd.read_excel('../jupyter-notebooks/asset_prices.xlsx',index_col='Date')

# PREPROCESS FINANCIAL RATIOS DATA, REPLACE STRINGS WITH FLOATS
ML_data = test_merge.map(data_preproc.convert_placeholder_text_to_num)

# ENSURE THE TWO DATAFRAMES CONTAINING FINANCIAL RATIOS (ML_DATA) AND RETURNS (TEST_ASSETS) HAVE THE SAME ASSETS/TICKERS
ML_final = data_preproc.filter_ratios_returns(ML_data,test_assets)
# print(ML_final.head())

# RESAMPLE THE RETURNS FROM MONTHLY TO QUARTERLY, THEN BFILL AND FFILL
asset_prices = test_assets # MAKE A COPY
asset_prices.index = pd.to_datetime(asset_prices.index)
asset_prices = asset_prices.resample('Q').last()
asset_prices = asset_prices.bfill(axis=1)
asset_prices = asset_prices.ffill(axis=1)


# 
test = data_preproc.FRatioMLdata(ML_final,asset_prices,sector=None,returns_lead_by=-1)
test.transform()
print(test.train.head())
#print(type(test.transform().head()))

#def main():
#    '''
#    Main code to execute.
#    '''
#    pass

#if __name__ == 'main':
#    main() # sys.argv[1]