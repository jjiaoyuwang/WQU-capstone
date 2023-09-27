import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# FUNCTIONS FOR CONVERTING BETWEEN TREND/RETURNS PREDICTION

def convert_returns_to_category(element):
    if element>= 0:
        element = 1
    if element < 0:
        element = 0
    return element

def convert_regression_to_classification(dataframe):
    '''
    Given a FRatioMLdata object i.e. [ratio_1 ... ratio_n returns], convert the returns column to:
    1 - if return >= 0
    0 - if return < 0
    '''

    df = dataframe.copy()

    df['Returns'] = df['Returns'].map(convert_returns_to_category)
    return df

def gen_train_test(dataframe,regression=True):
    '''
    Need to account for different cases of regression vs classification
    dataframe - 
    regression - 
    '''

    X = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    
    # scale the data
    data_scaler_x = StandardScaler()
    X = data_scaler_x.fit_transform(X.values)

    if regression is True:
        data_scaler_y = StandardScaler()
        y = data_scaler_y.fit_transform(y.values.reshape(-1,1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)
    return X_train, X_test, y_train, y_test