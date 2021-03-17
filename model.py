import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

if __name__ == "__main__":
    train_df = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/train.csv')
    test_df = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/test.csv')
    y = train_df['target']
    x = train_df.drop('target',axis=1)

    feature_name = x.columns.tolist()

    model = LGBMClassifier(random_state=27, n_estimators=830,num_leaves=454, max_depth=61,
                       learning_rate=0.006910869038433314, min_split_gain=0.00667926424629105, 
                       feature_fraction=0.3764303138879782, bagging_freq=8)

    stkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
    
    sample_submission = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/sample_submission.csv')
    probs = calc(x, y, model, stkfold, test_df)
    submission = pd.DataFrame({
        "client_id": sample_submission["client_id"],
        "target": probs['target']
    })
    submission.to_csv('/content/drive/MyDrive/competitions/recog-r1/submission.csv', index=False)