import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# IMPORT FUNCTIONS
# insert the following line if you need to import from outside of sample/
# sys.path.insert(0, '../sample')

import data_preproc
import ML_routines
import models


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

# transform the data into ML compatible format
test.transform()

# GENERATE REGRESSION AND CLASSIFICATION DATASETS WITH DETERMINISTIC SHUFFLE
data_rg = shuffle(test.train,random_state=0)
data_clf = ML_routines.convert_regression_to_classification(data_rg)
