import pandas as pd
import numpy as np
import statsmodels.api as sm

# IMPORT FUNCTIONS
import ML_routines

# ML - preprocessing
from sklearn.model_selection import GridSearchCV


# baseline models
from sklearn import linear_model

# models

from sklearn import tree
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn import ensemble


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

    svr = svm.SVR(kernel=svr_cv.best_params_['kernel'],C=svr_cv.best_params_['C'],epsilon=svr_cv.best_params_['epsilon']).fit(X_train,y_train)

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
                                   bootstrap_features=br_cv.best_params_['bootstrap_features']).fit(X_train,y_train)

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