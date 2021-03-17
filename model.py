import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from optuna import Trial
import gc
import optuna
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

def calc(X, y, model, cv, test_df):
    res=[]
    local_probs=pd.DataFrame()
    probs = pd.DataFrame()

    for i, (tdx, vdx) in enumerate(cv.split(X, y)):
        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
        model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_valid, y_valid)],
                 early_stopping_rounds=30, verbose=False)
        
        preds = model.predict_proba(X_valid)
        oof_predict = model.predict_proba(test_df)
        local_probs['fold_%i'%i] = oof_predict[:,1]
        res.append(roc_auc_score(y_valid, preds[:,1]))

    print('ROC AUC:', round(np.mean(res), 6))    
    local_probs['res'] = local_probs.mean(axis=1)
    probs['target'] = local_probs['res']
    
    return probs

def rfc(X, y, model, cv, test_df):
    res=[]
    local_probs=pd.DataFrame()
    probs = pd.DataFrame()

    for i, (tdx, vdx) in enumerate(cv.split(X, y)):
        print('*'*10)
        print(i)
        print('*'*10)
        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
        model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_valid, y_valid)],
                 early_stopping_rounds=30, verbose=2)
        
        preds = model.predict_proba(X_valid)
        oof_predict = model.predict_proba(test_df)
        print(oof_predict)
        local_probs['fold_%i'%i] = oof_predict[:,1]
        res.append(roc_auc_score(y_valid, preds[:,1]))

    print('ROC AUC:', round(np.mean(res), 6))    
    local_probs['res'] = local_probs.mean(axis=1)
    probs['target'] = local_probs['res']
    
    return probs

def fitBoost(trial,X, y):
    
    params={
      'n_estimators':trial.suggest_int('n_estimators', 0, 1000), 
    #   'max_depth':trial.suggest_int('max_depth', 2, 128),
    #   'min_child_weight':trail.suggest_int('min_child_weight',2,128),
    #   'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.15),
    #   'gamma':trail.suggest_int('gamma',0,10).
    #   'verbosity': -1,
    #   'subsample':trail.suggest_loguniform('subsample', 0.5,1.0)
    #   'random_state':seed
    }
    stkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model = XGBClassifier(**params)
    
    res=[]
    for i, (tdx, vdx) in enumerate(stkfold.split(X, y)):
        X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
        model.fit(X_train, y_train,)
                #  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                #  early_stopping_rounds=30, verbose=False)
        preds = model.predict_proba(X_valid)
        res.append(roc_auc_score(y_valid, preds[:,1]))
    
    err = np.mean(res)
    
    return model, err

if __name__ == "__main__":
    seed=47
    train_df = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/train.csv')
    test_df = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/test.csv')

    print(train_df.head())
    # train_df = train_df.reset_index()
    # test_df = test_df.reset_index()
    # train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # train_df.drop(columns=['delta_index_max_mean'],inplace=True)
    # test_df.drop(columns=['delta_index_max_mean'],inplace=True)
    # train_df.fillna(1,inplace=True)
    # test_df.fillna(1,inplace=True)

    y = train_df['target']
    x = train_df.drop('target',axis=1)

    def tune(trial:Trial):
        gc.collect()
        models=[]
        validScore=0
    
        model,log = fitBoost(trial,x,y)
        
        models.append(model)
        gc.collect()
        validScore+=log
        validScore/=len(models)
        
        return validScore

    # study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    # study.optimize(tune, timeout=60*60*2)



    # print(study.best_params)

    # exit()

    feature_name = x.columns.tolist()


    print(x.isnull().sum().sum())
    # model = LGBMClassifier(random_state=seed, n_estimators=830,num_leaves=454, max_depth=61,
    #                    learning_rate=0.006910869038433314, min_split_gain=0.00667926424629105, 
    #                    feature_fraction=0.3764303138879782, bagging_freq=8)

    # model = RandomForestClassifier(random_state=seed)
    # model = XGBClassifier( learning_rate =0.05,
    #                         n_estimators=2500,
    #                         max_depth=12,
    #                         min_child_weight=8,
    #                         gamma=5,
    #                         subsample=0.8,
    #                         colsample_bytree=0.8,
    #                         reg_alpha=0.03,
    #                         scale_pos_weight=1,
    #                         seed=27,
    #                         eval_metric='auc') 

    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1, eval_metric='auc')

    stkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    sample_submission = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/sample_submission.csv')
    probs = rfc(x, y, model, stkfold, test_df)
    submission = pd.DataFrame({
        "client_id": sample_submission["client_id"],
        "target": probs['target']
    })
    submission.to_csv('/content/drive/MyDrive/competitions/recog-r1/submission.csv', index=False)