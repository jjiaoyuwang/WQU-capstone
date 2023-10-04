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

# GET TRAINING AND TEST DATA GIVEN SECTOR AND LAG
def gen_train_test(ML_final, asset_prices, sector=None, returns_lead_by=-1):
    '''
    Generate regression and classification test/train data sets
    
    Input
    ML_final - df
    asset_prices - df
    sector - str (default None, otherwise choose Real Estate, Industrials or Consumer Discretionary)
    returns_lead_by - int (default -1, otherwise choose 0, 1 or 3)
    
    Output
    X_train, X_test, y_train, y_test - test/train split for regression models
    Xclf_train, Xclf_test, yclf_train, yclf_test - test/train split for classification models
    '''

    test = data_preproc.FRatioMLdata(ML_final,asset_prices,sector,returns_lead_by)

    # transform the data into ML compatible format
    test.transform()

    # generate Regression and Classification datasets with deterministic shuffle
    data_rg = shuffle(test.train,random_state=0)
    data_clf = ML_routines.convert_regression_to_classification(data_rg)

    # generate datasets for regression
    X_train, X_test, y_train, y_test =  ML_routines.gen_train_test(data_rg,regression=True)

    # generate datasets for classification
    Xclf_train, Xclf_test, yclf_train, yclf_test =  ML_routines.gen_train_test(data_clf,regression=False)

    return X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test





# TRAINING OR MODELS FROM FILE 

input_response = input('Would you like to train the models, type "Yes" (Warning: Can take a long time). Otherwise press "Enter": ')

def train_models():
    '''
    Run all models in models.py for the following lags: -1, 0, 1, 3 (corresponding to returns coincident, 1Q leading, 2Q leading and 4Q leading).

    Model cv objects, metrics, y_pred and model objects themselves are pickled and saved under model/LAG<lag number>.
    '''

    # Don't filter by sector, with 4 lags this is 4 models
    print("Training models...lag 0")
    path_prefix_to_models = "../models/LAG0Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=None, returns_lead_by=-1)    
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Training models...lag 1Q")
    path_prefix_to_models = "../models/LAG1Q/"  
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=None, returns_lead_by=0)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Training models...lag 2Q")
    path_prefix_to_models = "../models/LAG2Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=None, returns_lead_by=1)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Training models...lag 4Q")
    path_prefix_to_models = "../models/LAG4Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=None, returns_lead_by=3)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)


    # Filter by sector 3 sectors with 4 lags = 12 models
    # REAL ESTATE
    sector = 'Real Estate'
    print("Real Estate models...lag 0")
    path_prefix_to_models = "../models/REAL_ESTATE/LAG0Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=-1)    
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Real Estate models...lag 1Q")
    path_prefix_to_models = "../models/REAL_ESTATE/LAG1Q/"  
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=0)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Real Estate models...lag 2Q")
    path_prefix_to_models = "../models/REAL_ESTATE/LAG2Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=1)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Real Estate models...lag 4Q")
    path_prefix_to_models = "../models/REAL_ESTATE/LAG4Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=3)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    # INDUSTRIALS
    sector = 'Industrials'
    print("Industrials models...lag 0")
    path_prefix_to_models = "../models/INDUSTRIALS/LAG0Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=-1)    
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Industrials models...lag 1Q")
    path_prefix_to_models = "../models/INDUSTRIALS/LAG1Q/"  
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=0)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Industrials models...lag 2Q")
    path_prefix_to_models = "../models/INDUSTRIALS/LAG2Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=1)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Industrials models...lag 4Q")
    path_prefix_to_models = "../models/INDUSTRIALS/LAG4Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=3)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)


    # CONSUMER DISCRETIONARY
    sector = 'Consumer Discretionary'
    print("Consumer Discretionary models...lag 0")
    path_prefix_to_models = "../models/CONSUMER_DISC/LAG0Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=-1)    
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Consumer Discretionary models...lag 1Q")
    path_prefix_to_models = "../models/CONSUMER_DISC/LAG1Q/"  
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=0)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Consumer Discretionary models...lag 2Q")
    path_prefix_to_models = "../models/CONSUMER_DISC/LAG2Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=1)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)

    print("Consumer Discretionary models...lag 4Q")
    path_prefix_to_models = "../models/CONSUMER_DISC/LAG4Q/"
    X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=3)
    models.run_all_models(X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test,path_prefix_to_models=path_prefix_to_models)



if input_response == "Yes":
    
    train_models()
else:
    print("Models weren't trained")

# LOAD TRAINED MODELS INTO MEMORY
rg_models_list = [
    'LASSO',
    'ml_svr',
    'ml_dtr',
    'ml_abr',
    'ml_br',
    'ml_rfr',
    'ml_xgb',
    'dl_mlpr',
]

clf_models_list = [
    'Logistic',
    'ml_svc',
    'ml_dtc',
    'ml_abc',
    'ml_bc',
    'ml_rfc',
    'ml_xgbc',
    'dl_mlpc',
]

dict_of_model_dirs = {
    'ALLQ0': '../models/LAG0Q',
    'ALLQ1': '../models/LAG1Q',
    'ALLQ2': '../models/LAG2Q',
    'ALLQ4': '../models/LAG4Q',
    'REQ0': '../models/REAL_ESTATE/LAG0Q',
    'REQ1': '../models/REAL_ESTATE/LAG1Q',
    'REQ2': '../models/REAL_ESTATE/LAG2Q',
    'REQ4': '../models/REAL_ESTATE/LAG4Q',
    'INDQ0': '../models/INDUSTRIALS/LAG0Q',
    'INDQ1': '../models/INDUSTRIALS/LAG1Q',
    'INDQ2': '../models/INDUSTRIALS/LAG2Q',
    'INDQ4': '../models/INDUSTRIALS/LAG4Q',
    'CDQ0': '../models/CONSUMER_DISC/LAG0Q',
    'CDQ1': '../models/CONSUMER_DISC/LAG1Q',
    'CDQ2': '../models/CONSUMER_DISC/LAG2Q',
    'CDQ4': '../models/CONSUMER_DISC/LAG4Q',
}

for dataset in dict_of_model_dirs:
    print(dict_of_model_dirs[dataset])

    # get regression models
    rg_metrics_df = ML_routines.return_models_not_in_folder(rg_models_list,dict_of_model_dirs[dataset],1)

    # get classification models
    clf_metrics_df = ML_routines.return_models_not_in_folder(clf_models_list,dict_of_model_dirs[dataset],1)

    # write the results to disk

    ML_routines.from_models_return_metrics(rg_metrics_df,regression=True).to_excel(dict_of_model_dirs[dataset]+\
                                                                                   "/rg_summary.xlsx")

    ML_routines.from_models_return_metrics(clf_metrics_df,regression=False).to_excel(dict_of_model_dirs[dataset]+\
                                                                                   "/clf_summary.xlsx")


# to do: load the pickle files into memory, generate df and export to excel models for each lag. Additionally for regression problems, determine Diebold-Mariano test, although I suspect that this will not be as useful for my problem as it would've been for sequences. 


# Calculate Diebold-Mariano p-values

#X_train, X_test, y_train, y_test, Xclf_train, Xclf_test, yclf_train, yclf_test = gen_train_test(ML_final, asset_prices, sector=sector, returns_lead_by=-1)    

# train using the existing parameters in models.py. This should be (relatively quick). Then run on an aws ec2 instance with a finer grid for all models. 

# 
