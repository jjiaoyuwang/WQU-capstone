import pandas as pd
import numpy as np
import statsmodels.api as sm

# IMPORT FUNCTIONS
import ML_routines

# ML - preprocessing
from sklearn.model_selection import GridSearchCV


# baseline models
from sklearn import linear_model

# ML - models

from sklearn import tree
from sklearn.multioutput import RegressorChain
from sklearn import svm
from sklearn import ensemble

# NEURAL NETS (MLP)
import keras
import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

# fix random seed for reproducibility in keras models
tf.random.set_seed(0)


# LASSO REGRESSION

def lasso_run(X_train, X_test, y_train, y_test):
    '''
    Fit LASSO to training data and perform 5-fold CV (grid search). Return:
    [0] - model as an object
    [1] - metrics [R2_train, R2_test, MAE, MSE, MAPE]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''
    
    grid = {
        'alpha': list(np.logspace(-2, 3, 6))
    }

    reg_cv = GridSearchCV(estimator=linear_model.Lasso(), param_grid=grid,cv=5)
    reg_cv.fit(X_train, y_train)

    reg = linear_model.Lasso(alpha=reg_cv.best_params_['alpha']).fit(X_train,y_train)

    y_pred_scaled = reg.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # R^2 train error
    R2_train = reg.score(X_train, y_train)

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    # I don't need this line because it's given in the return_regress_metrics function.
    # print(f'R^2 error (test): {np.round(reg.score(X_test, y_test),5)}')

    metrics = ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return reg_cv, metrics, y_pred, reg

def logistic_run(X_train, X_test, y_train, y_test):
    '''
    Fit Logistic Regression model to training data and perform 5-fold CV (grid search). Return:
    [0] - model as an object
    [1] - metrics [AS train, AS test, f1, PS]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''
    
    grid = [
        {
        'penalty': ['l1', 'l2'],
        'C': list(np.logspace(-2, 3, 6)),
            'solver':['saga']
        
    },
        {
         'penalty': ['elasticnet'],
            'C': list(np.logspace(-2, 3, 6)),
            'l1_ratio': list(np.linspace(0,1,5)),
            'solver':['saga']
        }
    ]

    log_cv = GridSearchCV(estimator=linear_model.LogisticRegression(), param_grid=grid,cv=5)
    log_cv.fit(X_train, y_train) # fit(X_train,np.ravel(y_train))

    log = linear_model.LogisticRegression(penalty=log_cv.best_params_['penalty'],\
                                          C=log_cv.best_params_['C'],l1_ratio=log_cv.best_params_['l1_ratio'],\
                                         solver='saga').fit(X_train,y_train)

    y_pred_scaled = log.predict(X_test)
    y_pred = y_pred_scaled

    # Accuracy score on training set
    AS_train = log.score(X_train, y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return log_cv, metrics, y_pred, log

def SVR_run(X_train, X_test, y_train, y_test):
    '''
    Fit SVM regression to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [R2_train, R2_test, MAE, MSE, MAPE]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'kernel': ['linear','poly','rbf','sigmoid'],
        'C': list(np.logspace(-2, 3, 6)), 
        'epsilon': [0.01,0.1,1,10]
    }

    svr_cv = GridSearchCV(estimator=svm.SVR(), param_grid=grid,cv=5,n_jobs=4)
    svr_cv.fit(X_train, np.ravel(y_train))

    svr = svm.SVR(kernel=svr_cv.best_params_['kernel'],C=svr_cv.best_params_['C'],epsilon=svr_cv.best_params_['epsilon']).fit(X_train,np.ravel(y_train))

    y_pred_scaled = svr.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # R^2 train error
    R2_train = svr.score(X_train, y_train)

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    metrics =ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return svr_cv, metrics, y_pred, svr

def DTR_run(X_train, X_test, y_train, y_test):
    '''
    Fit Decision Tree regression to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [R2_train, R2_test, MAE, MSE, MAPE]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
        'splitter': ['best','random'],
        'max_features': ['sqrt', 'log2',None],
        'max_depth' : [3,4,5,6,7,8, None],
        'ccp_alpha': list(np.logspace(-2, 3, 6)),
    }

    dtr_cv = GridSearchCV(estimator=tree.DecisionTreeRegressor(), param_grid=grid,cv=5,n_jobs=4)
    dtr_cv.fit(X_train, y_train)#np.ravel(y_train))

    dtr = tree.DecisionTreeRegressor(criterion=dtr_cv.best_params_['criterion'],\
                                     splitter=dtr_cv.best_params_['splitter'],\
                                     max_features=dtr_cv.best_params_['max_features'],\
                                     max_depth=dtr_cv.best_params_['max_depth'],\
                                     ccp_alpha=dtr_cv.best_params_['ccp_alpha']).fit(X_train,y_train)

    y_pred_scaled = dtr.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # R^2 train error
    R2_train = dtr.score(X_train, y_train)

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    metrics = ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return dtr_cv, metrics, y_pred, dtr

# added to models.py

def BR_run(X_train, X_test, y_train, y_test):
    '''
    Fit Bagging Regressor using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [R2_train, R2_test, MAE, MSE, MAPE]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'estimator': [None], # None = DecisionTreeRegressor with max_depth=3
        'n_estimators': list(np.arange(1,50,5)),
        'max_samples': list(np.arange(1,50,5)),
        'max_features': [1.0], # default, I don't want to mess with the bootstrapping for the moment
        'bootstrap': [True], # default, I don't want to mess with the bootstrapping for the moment
        'bootstrap_features': [False] # default, I don't want to mess with the bootstrapping for the moment
    }

    br_cv = GridSearchCV(estimator=ensemble.BaggingRegressor(), param_grid=grid,cv=5,n_jobs=4)
    br_cv.fit(X_train, np.ravel(y_train))

    br = ensemble.BaggingRegressor(estimator=br_cv.best_params_['estimator'],\
                                   n_estimators=br_cv.best_params_['n_estimators'],\
                                   max_samples=br_cv.best_params_['max_samples'],\
                                   max_features=br_cv.best_params_['max_features'],\
                                   bootstrap=br_cv.best_params_['bootstrap'],\
                                   bootstrap_features=br_cv.best_params_['bootstrap_features']).fit(X_train,np.ravel(y_train))

    y_pred_scaled = br.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # R^2 train error
    R2_train = br.score(X_train, y_train)

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    metrics = ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return br_cv, metrics, y_pred, br

def RFR_run(X_train, X_test, y_train, y_test):
    '''
    Fit Random Forest Regressor using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [R2_train, R2_test, MAE, MSE, MAPE]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'n_estimators': list(np.arange(1,50,5)),
        'criterion': ['squared_error','friedman_mse','absolute_error','poisson'], # most of these parameters are the same as for decision tree regressor
        'max_depth': [3,4,5,6,7,8, None],
        'max_features': ['sqrt', 'log2',None],
        'max_samples': [None],
        'bootstrap': [True] # default, I don't want to mess with the bootstrapping for the moment
    }

    rfr_cv = GridSearchCV(estimator=ensemble.RandomForestRegressor(), param_grid=grid,cv=5,n_jobs=4)
    rfr_cv.fit(X_train, np.ravel(y_train))

    rfr = ensemble.RandomForestRegressor(n_estimators=rfr_cv.best_params_['n_estimators'],\
                                         criterion=rfr_cv.best_params_['criterion'],\
                                         max_depth=rfr_cv.best_params_['max_depth'],\
                                         max_samples=rfr_cv.best_params_['max_samples'],\
                                         max_features=rfr_cv.best_params_['max_features'],\
                                         bootstrap=rfr_cv.best_params_['bootstrap']).fit(X_train,y_train)

    y_pred_scaled = rfr.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # R^2 train error
    R2_train = rfr.score(X_train, y_train)

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    metrics = ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return rfr_cv, metrics, y_pred, rfr

def ABR_run(X_train, X_test, y_train, y_test):
    '''
    Fit Ada Boost Regressor using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [R2_train, R2_test, MAE, MSE, MAPE]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'estimator': [None], # None = DecisionTreeRegressor with max_depth=3
        'n_estimators': list(np.arange(1,201,20)),
        'learning_rate': list(np.arange(0,200,20))
    }

    abr_cv = GridSearchCV(estimator=ensemble.AdaBoostRegressor(), param_grid=grid,cv=5,n_jobs=4)
    abr_cv.fit(X_train, np.ravel(y_train))

    abr = ensemble.AdaBoostRegressor(estimator=abr_cv.best_params_['estimator'],\
                                     n_estimators=abr_cv.best_params_['n_estimators'],\
                                     learning_rate=abr_cv.best_params_['learning_rate']).fit(X_train,np.ravel(y_train))

    y_pred_scaled = abr.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # R^2 train error
    R2_train = abr.score(X_train, y_train)

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    metrics = ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return abr_cv, metrics, y_pred, abr

def XGB_run(X_train, X_test, y_train, y_test):
    '''
    Fit Gradient Boosting Regressor using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [R2_train, R2_test, MAE, MSE, MAPE]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = { 
        #'loss': ['squared_error','absolute_error','huber'],
        'learning_rate': list(np.logspace(-1, 1, 3)),
        'n_estimators': list(np.arange(1,201,25)), # same for 
        #'criterion': ['squared_error','friedman_mse'], # most of these parameters are the same as for decision tree regressor
        'max_depth': [1,3,5,7, None],
        'max_features': ['sqrt', 'log2',None]
    }

    xgb_cv = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(), param_grid=grid,cv=5,n_jobs=4)
    xgb_cv.fit(X_train, np.ravel(y_train))

    xgb = ensemble.GradientBoostingRegressor(learning_rate=xgb_cv.best_params_['learning_rate'],\
                                             n_estimators=xgb_cv.best_params_['n_estimators'],\
                                             #criterion=xgb_cv.best_params_['criterion'],\
                                             #loss=xgb_cv.best_params_['loss'],\
                                             max_depth=xgb_cv.best_params_['max_depth'],\
                                             max_features=xgb_cv.best_params_['max_features']).fit(X_train,np.ravel(y_train))

    y_pred_scaled = xgb.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # R^2 train error
    R2_train = xgb.score(X_train, y_train)

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    metrics = ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return xgb_cv, metrics, y_pred, xgb

def SVC_run(X_train, X_test, y_train, y_test):
    '''
    Fit SVC to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''
    # define grid of parameters to search
    grid = {
        'kernel': ['linear','poly','rbf','sigmoid'],
        'C': list(np.logspace(-2, 3, 6)), 
        'degree': [3]
    }
    
    svc_cv = GridSearchCV(estimator=svm.SVC(), param_grid=grid,cv=5,n_jobs=4)
    svc_cv.fit(X_train,np.ravel(y_train))

    svc = svm.SVC(C=svc_cv.best_params_['C'], kernel=svc_cv.best_params_['kernel']).fit(X_train,y_train)

    # get predicted values (out of sample performance)
    y_pred_scaled = svc.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # Accuracy score on training set
    AS_train = svc.score(X_train, y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return svc_cv, metrics, y_pred, svc

def DTC_run(X_train, X_test, y_train, y_test):
    '''
    Fit Decision Tree Classifier to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''
    grid = {
        'criterion': ['gini','entropy','log_loss'],
        'splitter': ['best','random'],
        'max_features': ['sqrt', 'log2',None],
        'max_depth' : [3,4,5,6,7,8, None],
        'ccp_alpha': list(np.logspace(-2, 3, 6)),
    }
    
    dtc_cv = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=grid,cv=5,n_jobs=4)
    dtc_cv.fit(X_train,y_train)

    dtc = tree.DecisionTreeClassifier(criterion=dtc_cv.best_params_['criterion'],\
                                      splitter=dtc_cv.best_params_['splitter'],\
                                      ccp_alpha=dtc_cv.best_params_['ccp_alpha'],\
                                      max_depth=dtc_cv.best_params_['max_depth'],\
                                      max_features=dtc_cv.best_params_['max_features']).fit(X_train,y_train)
    
    # get predicted values (out of sample performance)
    y_pred_scaled = dtc.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # Accuracy score on training set
    AS_train = dtc.score(X_train, y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return dtc_cv, metrics, y_pred, dtc

def RFC_run(X_train, X_test, y_train, y_test):
    '''
    Fit Random Forest Classifier using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'n_estimators': list(np.arange(1,200,5)),
        'criterion': ['gini','entropy','log_loss'], # most of these parameters are the same as for decision tree regressor
        'max_depth': [3,4,5,6,7,8, None],
        'max_features': ['sqrt', 'log2',None],
        'max_samples': [None], # default
        'bootstrap': [True] # default, I don't want to mess with the bootstrapping for the moment
    }

    rfc_cv = GridSearchCV(estimator=ensemble.RandomForestClassifier(), param_grid=grid,cv=5,n_jobs=4)
    rfc_cv.fit(X_train, np.ravel(y_train))

    rfc= ensemble.RandomForestClassifier(n_estimators=rfc_cv.best_params_['n_estimators'],\
                                         criterion=rfc_cv.best_params_['criterion'],\
                                         max_depth=rfc_cv.best_params_['max_depth'],\
                                         max_samples=rfc_cv.best_params_['max_samples'],\
                                         max_features=rfc_cv.best_params_['max_features'],\
                                         bootstrap=rfc_cv.best_params_['bootstrap']).fit(X_train,y_train)

    # get predicted values (out of sample performance)
    y_pred_scaled = rfc.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # Accuracy score on training set
    AS_train = rfc.score(X_train, y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return rfc_cv, metrics, y_pred, rfc

def BC_run(X_train, X_test, y_train, y_test):
    '''
    Fit Bagging Classifier using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'estimator': [None], # None = DecisionTreeClassifier with max_depth=3
        'n_estimators': list(np.arange(1,50,5)),
        'max_samples': list(np.arange(1,50,5)),
        'max_features': [1.0], # default, I don't want to mess with the bootstrapping for the moment
        'bootstrap': [True], # default, I don't want to mess with the bootstrapping for the moment
        'bootstrap_features': [False] # default, I don't want to mess with the bootstrapping for the moment
    }

    bc_cv = GridSearchCV(estimator=ensemble.BaggingClassifier(), param_grid=grid,cv=5,n_jobs=4)
    bc_cv.fit(X_train, np.ravel(y_train))

    bc = ensemble.BaggingClassifier(estimator=bc_cv.best_params_['estimator'],\
                                   n_estimators=bc_cv.best_params_['n_estimators'],\
                                   max_samples=bc_cv.best_params_['max_samples'],\
                                   max_features=bc_cv.best_params_['max_features'],\
                                   bootstrap=bc_cv.best_params_['bootstrap'],\
                                   bootstrap_features=bc_cv.best_params_['bootstrap_features']).fit(X_train,y_train)

    y_pred_scaled = bc.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # Accuracy score on training set
    AS_train = bc.score(X_train, y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return bc_cv, metrics, y_pred, bc

def ABC_run(X_train, X_test, y_train, y_test):
    '''
    Fit Ada Boost Classifier using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = {
        'estimator': [None], # None = DecisionTreeRegressor with max_depth=1
        'n_estimators': list(np.arange(1,201,20)),
        'learning_rate': list(np.arange(0,200,20)),
        'algorithm': ['SAMME.R'] # default
    }

    abc_cv = GridSearchCV(estimator=ensemble.AdaBoostClassifier(), param_grid=grid,cv=5,n_jobs=4)
    abc_cv.fit(X_train, np.ravel(y_train))

    abc = ensemble.AdaBoostClassifier(estimator=abc_cv.best_params_['estimator'],\
                                     n_estimators=abc_cv.best_params_['n_estimators'],\
                                     learning_rate=abc_cv.best_params_['learning_rate'],\
                                     algorithm=abc_cv.best_params_['algorithm']).fit(X_train,y_train)

    y_pred_scaled = abc.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # Accuracy score on training set
    AS_train = abc.score(X_train, y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return abc_cv, metrics, y_pred, abc

def XGBC_run(X_train, X_test, y_train, y_test):
    '''
    Fit Gradient Boosting Classifier using decision trees as the estimator to training data and perform 5-fold CV (grid search). Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''

    grid = { 
        #'loss': ['log_loss','exponential'],
        'learning_rate': list(np.logspace(-1, 1, 3)),
        'n_estimators': list(np.arange(1,201,25)), # same for 
        #'criterion': ['squared_error','friedman_mse'], # most of these parameters are the same as for decision tree regressor
        'max_depth': [1,3,5,7, None],
        'max_features': ['sqrt', 'log2',None]
    }

    xgbc_cv = GridSearchCV(estimator=ensemble.GradientBoostingClassifier(), param_grid=grid,cv=5,n_jobs=4)
    xgbc_cv.fit(X_train, np.ravel(y_train))

    xgbc = ensemble.GradientBoostingClassifier(#loss=xgbc_cv.best_params_['loss'],\
                                             learning_rate=xgbc_cv.best_params_['learning_rate'],\
                                             n_estimators=xgbc_cv.best_params_['n_estimators'],\
                                             # criterion=xgbc_cv.best_params_['criterion'],\
                                             max_depth=xgbc_cv.best_params_['max_depth'],\
                                             max_features=xgbc_cv.best_params_['max_features']).fit(X_train,y_train)

    y_pred_scaled = xgbc.predict(X_test)
    y_pred = y_pred_scaled#data_scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

    # Accuracy score on training set
    AS_train = xgbc.score(X_train, y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return xgbc_cv, metrics, y_pred, xgbc

def MLP_clf(hidden_layers, neurons, dropout,optimizer='adam',activation='relu'):
    model = keras.models.Sequential()

    # define an input layer with dim 8 (8 financial ratios)
    model.add(keras.layers.Input(shape=(8,)))
    
    for i in range(hidden_layers):
        model.add(keras.layers.Dense(units=neurons, activation=activation))
        model.add(keras.layers.Dropout(dropout))

    # for classification problem, the final layer must output a sigmoid
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # obsolete, passed the optimiser to get_clf, remove this line going forward
    # comment out the line below if you're doing CV on the optimizer
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model


def MLP_clf_run(X_train, X_test, y_train, y_test):
    '''
    Fit an MLP NN classifier to training data and perform 5-fold CV (grid search). 
    
    This is a sequential model from the keras package. scikeras package is used to interface the keras
    objects with sklearn so that GridSearchCV can be performed. 
    
    Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''
    
    clf = KerasClassifier(model=MLP_clf,verbose=False,)

    grid = {
        #'optimizer__learning_rate': [0.05, 0.1], # adam adapts its learning_rate automatically.
        'model__hidden_layers': [1, 2, 3],
        'model__neurons': [32],# 64, 128],
        'model__dropout': [0, 0.5],
        'model__activation': ['relu','softmax', 'tanh', 'sigmoid', 'linear'],
        'optimizer': ['Adam'],# 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
        #'batch_size': [10,20],#40,60,80,100],
        #'epochs': [10]#,50,100]
    }

    NN_cv = GridSearchCV(clf, grid, scoring='accuracy',cv=5, n_jobs=-1, verbose=False)
    NN_cv.fit(X_train, y_train)

    NN = KerasClassifier(model=MLP_clf,\
                         activation=NN_cv.best_params_['model__activation'],\
                         dropout=NN_cv.best_params_['model__dropout'],\
                         hidden_layers=NN_cv.best_params_['model__hidden_layers'],\
                         neurons=NN_cv.best_params_['model__neurons'],\
                         optimizer=NN_cv.best_params_['optimizer'],\
                         #learning_rate=NN_cv.best_params_['optimizer__learning_rate'],\
                         verbose=False,).fit(X_train,y_train)

    y_pred_scaled = NN.predict(X_test)
    y_pred = y_pred_scaled

    # Accuracy score on training set
    AS_train = NN.score(X_train,y_train)

    print(f'Accuracy Score (train): {np.round(AS_train,5)}')

    metrics = ML_routines.return_class_metrics(y_test,y_pred)
    metrics.insert(0,AS_train)

    return NN_cv, metrics, y_pred, NN

def MLP_rg(hidden_layers, neurons, dropout,optimizer='adam',activation='relu'):
    model = keras.models.Sequential()

    # define an input layer with dim 8 (8 financial ratios)
    model.add(keras.layers.Input(shape=(8,)))
    
    for i in range(hidden_layers):
        model.add(keras.layers.Dense(units=neurons, activation=activation))
        model.add(keras.layers.Dropout(dropout))

    # for classification problem, the final layer must output a sigmoid
    model.add(keras.layers.Dense(1))

    model.compile(loss="mse", optimizer=optimizer, metrics=[KerasRegressor.r_squared])
    return model


def MLP_rg_run(X_train, X_test, y_train, y_test):
    '''
    Fit an MLP NN regressor to training data and perform 5-fold CV (grid search). 
    
    This is a sequential model from the keras package. scikeras package is used to interface the keras
    objects with sklearn so that GridSearchCV can be performed. 
    
    Return:
    [0] - model_cv as an object
    [1] - metrics [AS_train, AS_test, F1, PS, AUC]
    [2] - predicted values on test set y_test
    [3] - model as an object
    '''
    
    rg = KerasRegressor(model=MLP_rg,verbose=False,)

    grid = {
        #'optimizer__learning_rate': [0.05, 0.1], # adam adapts its learning_rate automatically.
        'model__hidden_layers': [1, 2, 3],
        'model__neurons': [32],# 64, 128],
        'model__dropout': [0, 0.5],
        'model__activation': ['relu','softmax', 'tanh', 'sigmoid', 'linear'],
        'optimizer': ['Adam'],# 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
        #'batch_size': [10,20],#40,60,80,100],
        #'epochs': [10]#,50,100]
    }

    NN_cv = GridSearchCV(rg, grid,cv=5, n_jobs=-1, verbose=False)
    NN_cv.fit(X_train, np.ravel(y_train))

    NN = KerasRegressor(model=MLP_rg,\
                         activation=NN_cv.best_params_['model__activation'],\
                         dropout=NN_cv.best_params_['model__dropout'],\
                         hidden_layers=NN_cv.best_params_['model__hidden_layers'],\
                         neurons=NN_cv.best_params_['model__neurons'],\
                         optimizer=NN_cv.best_params_['optimizer'],\
                         #learning_rate=NN_cv.best_params_['optimizer__learning_rate'],\
                         verbose=False,).fit(X_train,np.ravel(y_train))

    y_pred_scaled = NN.predict(X_test)
    y_pred = y_pred_scaled

    # Accuracy score on training set
    R2_train = NN.score(X_train,np.ravel(y_train))

    print(f'R^2 error (train): {np.round(R2_train,5)}')

    metrics = ML_routines.return_regress_metrics(y_test,y_pred)
    metrics.insert(0,R2_train)

    return NN_cv, metrics, y_pred, NN


def run_all_models(X_train, X_test, y_train, y_test,\
    Xclf_train, Xclf_test, yclf_train, yclf_test,\
    path_prefix_to_models='../models/proto/'):

    '''
    '''

    # Regression models 
    test_lasso = lasso_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(test_lasso, path_prefix_to_models+"LASSO.pickle")

    ml_svr = SVR_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(ml_svr, path_prefix_to_models+"ml_svr.pickle") 

    ml_dtr = DTR_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(ml_dtr, path_prefix_to_models+"ml_dtr.pickle")

    ml_abr = ABR_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(ml_abr, path_prefix_to_models+"ml_abr.pickle")

    ml_br = BR_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(ml_br, path_prefix_to_models+"ml_br.pickle")

    ml_rfr = RFR_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(ml_rfr, path_prefix_to_models+"ml_rfr.pickle")

    ml_xgb = XGB_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(ml_xgb, path_prefix_to_models+"ml_xgb.pickle")

    dl_mlpr = MLP_rg_run(X_train, X_test, y_train, y_test)
    ML_routines.persist_model(dl_mlpr,  path_prefix_to_models+"dl_mlpr.pickle") 

    # Classification models

    test_logistic = logistic_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(test_logistic, path_prefix_to_models+"Logistic.pickle")

    ml_svc = SVC_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(ml_svc, path_prefix_to_models+"ml_svc.pickle")

    ml_dtc = DTC_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(ml_dtc, path_prefix_to_models+"ml_dtc.pickle")

    ml_rfc = RFC_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(ml_rfc, path_prefix_to_models+"ml_rfc.pickle")

    ml_bc = BC_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(ml_bc, path_prefix_to_models+"ml_bc.pickle")

    ml_abc = ABC_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(ml_abc, path_prefix_to_models+"ml_abc.pickle")

    ml_xgbc = XGBC_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(ml_xgbc, path_prefix_to_models+"ml_xgbc.pickle")

    dl_mlpc = MLP_clf_run(Xclf_train, Xclf_test, yclf_train, yclf_test)
    ML_routines.persist_model(dl_mlpc, path_prefix_to_models+"dl_mlpc.pickle")

