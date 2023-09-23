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
print(ML_data.head())
#def main():
#    '''
#    Main code to execute.
#    '''
#    pass

#if __name__ == 'main':
#    main() # sys.argv[1]