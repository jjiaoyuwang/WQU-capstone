# a module to generate plots used in the final report

import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import seaborn as sns

# IMPORT FUNCTIONS
# insert the following line if you need to import from outside of sample/
# sys.path.insert(0, '../sample')

import data_preproc
import ML_routines
#import models


# LOAD FINANCIAL RATIOS AND ASSET PRICES
test_merge = pd.read_excel('../jupyter-notebooks/test_manual.xlsx')
test_merge = test_merge.loc[:, test_merge.columns != 'Unnamed: 0']
test_assets = pd.read_excel('../jupyter-notebooks/asset_prices.xlsx',index_col='Date')

# PREPROCESS FINANCIAL RATIOS DATA, REPLACE STRINGS WITH FLOATS
#ML_data = test_merge.map(data_preproc.convert_placeholder_text_to_num)

# ENSURE THE TWO DATAFRAMES CONTAINING FINANCIAL RATIOS (ML_DATA) AND RETURNS (TEST_ASSETS) HAVE THE SAME ASSETS/TICKERS
#ML_final = data_preproc.filter_ratios_returns(ML_data,test_assets)
# print(ML_final.head())

# RESAMPLE THE RETURNS FROM MONTHLY TO QUARTERLY, THEN BFILL AND FFILL
#asset_prices = test_assets # MAKE A COPY
#asset_prices.index = pd.to_datetime(asset_prices.index)
#asset_prices = asset_prices.resample('Q').last()
#asset_prices = asset_prices.bfill(axis=1)
#asset_prices = asset_prices.ffill(axis=1)

#test = data_preproc.FRatioMLdata(ML_final,asset_prices,sector=None,returns_lead_by=-1)

#test.transform()



def get_filtered_df(dataframe,sector,financial_ratio):
    '''
    Filter the dataframe for a given financial ratio and sector
    '''
    
    filtered_df = dataframe.loc[dataframe['Sector'] == sector]
    return filtered_df.loc[:,filtered_df.columns.str.contains(financial_ratio)]

def get_length_of_dataframe(dataframe, sector, financial_ratio):
    '''
    For a dataframe containing financial ratios, return the number of rows after all NaN's are dropped
    '''
    filtered_df = dataframe.loc[dataframe['Sector'] == sector]
    return len(filtered_df.loc[:,filtered_df.columns.str.contains(financial_ratio)].dropna())

print(test_merge)

list_of_financial_ratio_strings = ['EV', 'FCF', 'EBITDA', 'Revenue', 'ROE', 'Gross-Profit-Margin', 'Quick-Ratio','Debt / Equity']
list_of_sectors = list(set(test_merge['Sector'].values))

df_length_mat = np.zeros([len(list_of_financial_ratio_strings),len(list_of_sectors)])
for i in range(len(list_of_financial_ratio_strings)):
    for j in range(len(list_of_sectors)):
        df_length_mat[i,j] = get_length_of_dataframe(test_merge,list_of_sectors[j],list_of_financial_ratio_strings[i])

heatmap_test = pd.DataFrame(df_length_mat,columns=list_of_sectors,index=list_of_financial_ratio_strings)


sns.set(rc={'figure.figsize':(14,10)})
sns.heatmap(heatmap_test,annot=True, cmap='Blues', fmt='.3g')
fig = sns.heatmap(heatmap_test,annot=True, cmap='Blues', fmt='.3g').get_figure()
fig.savefig("out.png") 
