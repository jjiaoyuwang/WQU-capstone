import pandas as pd
import numpy as np
import statsmodels.api as sm

# ML - preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# metrics - classification
from sklearn.metrics import PredictionErrorDisplay, accuracy_score, f1_score, precision_score, roc_auc_score

# metrics - regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# model comparison
from dieboldmariano import dm_test

# model persistence
import pickle



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

# USEFUL FUNCTIONS

def persist_model(model,filename):
    '''
    Given an sklearn model object, save the resulting model to file filename.
    Store models in directory ../models.
    '''

    with open(filename, 'wb') as f:
        pickle.dump(model,f)
    
    # only useful to load models for testing
    #with open('../models/test_lasso.pickle','rb') as f:
    #    test_lasso_2 = pickle.load(f)

def return_regress_metrics(y_test,y_pred):
    '''
    Given a regression type problem model (sklearn), return the following metrics as a list:
    Mean Absolute Error (MAE)
    Mean Squared Error (MSE)
    R^2 error
    Mean Absolute Percentage Error (MAPE)
    '''

    # Mean Absolute Error (MAE)
    MAE = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error (MAE): {np.round(MAE, 2)}')

    # Mean Squared Error (MSE)
    MSE = mean_squared_error(y_test,y_pred)
    print(f'Mean Squared Error (MSE): {np.round(MSE, 2)}')

    # R^2 error
    R2 = r2_score(y_test, y_pred)
    print(f'R^2 error (test): {np.round(R2, 2)}')

    # Mean Absolute Percentage Error (MAPE)
    MAPE = mean_absolute_percentage_error(y_test,y_pred)
    print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)}')

    return [R2, MAE, MSE, MAPE]

def return_class_metrics(y_test,y_pred):
    '''
    Given a regression type problem model (sklearn), return the following metrics as a list:
    F1 Score
    Precision Score
    AUC
    Accuracy Score
    '''
    
    # Accuracy Score
    AS = accuracy_score(y_test, y_pred)
    print(f'Accuracy Score (test): {np.round(AS, 2)}')
    
    # F1 score (best 1 - worst 0)
    f1 = f1_score(y_test,y_pred)
    print(f'F1: {np.round(f1, 2)}')

    # precision_score (the ability of the classifier not to label as positive a sample that is negative, best 1 - worst 0)
    PS = precision_score(y_test,y_pred)
    print(f'Precision Score: {np.round(PS, 2)}')

    # roc_auc_score
    AUC = roc_auc_score(y_test,y_pred)
    print(f'Reciever Operating Curve (Area Under Curve): {np.round(AUC, 2)}')

    return [AS, f1, PS, AUC]

def from_models_return_metrics(models_dict,regression):
    '''
    Input - a dictionary with keys as model names / descr and values as ML metrics, 
          - regression = True [R2_train, R2_test, MAE, MSE, MAPE], False [AS train, AS test, f1, PS]
    Output - dataframe summarising ML metrics with column names defined according to regression
    '''
    if regression is True:
        df = pd.DataFrame.from_dict(models_dict,orient='index',columns=['R^2 Score Train', 'R^2 Score Test', 'MAE', 'MSE', 'MAPE'])
    else:
        df = pd.DataFrame.from_dict(models_dict,orient='index',columns=['Accuracy Train','Accuracy Test', 'F1 Score', 'Precision Score', 'ROC AUC'])
    return df

def from_models_return_diebold_mariano(models_dict,y_test):
    '''
    Only applies to regression type models.
    Input - a dictionary with keys as model names / descr and values as y_pred. 
          - y_test, the test set common to all models
    Output - dataframe showing p-value of DM test
    '''
    labels = list(models_dict.keys())
    X = list(models_dict.values())
    Y = X

    result = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            if i != j:
                result[i,j] = dm_test(y_test, X[i], Y[j], one_sided=False)[1][0]
            else:
                pass
                
    df = pd.DataFrame(result)
    df.index = labels
    df.columns = labels
    return df

