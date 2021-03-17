import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import time

def feat(cl, inv):
    cl['client_catg'] = cl['client_catg'].astype('category')
    cl['district'] = cl['disrict'].astype('category')
    cl['region'] = cl['region'].astype('category')
    cl['region_group'] = cl['region'].apply(lambda x:100 if x<100 else 300 if x>300 else 200)
    cl['creation_date'] = pd.to_datetime(cl['creation_date'])
    cl['coop_time'] = (2019-cl['creation_date'].dt.year)*12 - cl['creation_date'].dt.month

    inv['counter_type'] = inv['counter_type'].map({'ELEC':1,'GAZ':0})
    inv['counter_statue'] = inv['counter_statue'].map({0:0,
                                                        1:1,
                                                        2:2,
                                                        3:3,
                                                        4:4,
                                                        5:5,
                                                        769:5,
                                                        '0':0,
                                                        '1':1,
                                                        '4':4,
                                                        '5':5,
                                                        'A':0,
                                                        618:5,
                                                        269375:5,
                                                        46:5,
                                                        420:5,
                                                        })
    inv['invoice_date'] = pd.to_datetime(inv['invoice_date'], dayfirst=True)
    inv['invoice_month'] = inv['invoice_date'].dt.month
    inv['invoice_year'] = inv['invoice_date'].dt.year
    inv['is_weekday'] = ((pd.DatetimeIndex(inv.invoice_date).dayofweek)//5 ==1).astype(float)
    inv['delta_index'] = inv['new_index'] - inv['old_index']
    return cl, inv

if __name__ == '__main__':
    # invoice_test = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/test_invoice.csv',
    #                         low_memory=False)
    # invoice_train = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/train_invoice.csv',
    #                         low_memory=False)
    # client_test = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/client_test.csv',
    #                         low_memory=False)
    # client_train = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/client_train.csv',
    #                         low_memory=False)
    # sample_submission = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/sample_submission.csv',
    #                         low_memory=False)
    # tr_cl, tr_inv = feat(client_train, invoice_train)
    # ts_cl ,ts_inv = feat(client_test, invoice_test)

    # tr_cl.to_csv('tr_cl.csv')
    # tr_inv.to_csv('tr_inv.csv')
    # ts_cl.to_csv('ts_cl.csv')
    # ts_inv.to_csv('ts_inv.csv')
    