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

def agg_feature(inv, cl, agg_stat):
    inv['invoice_date'] = pd.to_datetime(inv['invoice_date'], dayfirst=True)
    inv['delta_time'] = inv.sort_values(['client_id','invoice_date']).groupby('client_id')['invoice_date'].diff().dt.days.reset_index(drop=True)
    # print([agg_stat['delta_time']])
    agg_trans = inv.groupby('client_id')[agg_stat+['delta_time']].agg(['mean','std','min','max'])
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.sort_values()]
    agg_trans.reset_index(inplace=True)
    df = inv.groupby('client_id').size().reset_index(name='transactions_count')
    agg_trans = pd.merge(df, agg_trans, on='client_id',how='left')
    weekday_avg = inv.groupby('client_id')[['is_weekday']].agg(['mean'])
    weekday_avg.columns = ['_'.join(col).strip() for col in weekday_avg.columns.values]
    weekday_avg.reset_index(inplace=True)
    cl = pd.merge(cl, weekday_avg, on='client_id', how='left')
    full_df = pd.merge(cl, agg_trans, on='client_id', how='left')
    full_df['invoice_per_cooperation'] = full_df['transactions_count'] / full_df['coop_time']
    print(full_df.head())
    for col in agg_stat:
        full_df[col+'_range'] = full_df[col+'_max']-full_df[col+'_min']
        full_df[col+'_max_mean'] = full_df[col+'_max']/full_df[col+'_mean']

    full_df = full_df.drop(['client_id'],axis=1,inplace=True)
    full_df = full_df.drop(['creation_date'],axis=1,inplace=True)
    print(f"num columns is {len(full_df.columns)}")
    return full_df

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

    #tr_cl.to_csv('tr_cl.csv')
    # tr_inv.to_csv('tr_inv.csv')
    # ts_cl.to_csv('ts_cl.csv')
    # ts_inv.to_csv('ts_inv.csv')

    tr_cl = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/tr_cl.csv')
    tr_inv = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/tr_inv.csv')
    ts_cl = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/ts_cl.csv')
    ts_inv = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/ts_inv.csv')
    
    agg_stat_columns = [
                        'tarif_type',
                        'counter_number',
                        'counter_statue',
                        'counter_code',
                        'reading_remarque',
                        'consommation_level_1',
                        'consommation_level_2',
                        'consommation_level_3',
                        'consommation_level_4',
                        'old_index',
                        'new_index',
                        'months_number',
                        'counter_type',
                        'invoice_month',
                        'invoice_year',
                        'delta_index'
                        ]

    train_df = agg_feature(tr_inv,tr_cl,agg_stat_columns)
    test_df = agg_feature(ts_inv,ts_cl,agg_feature)

    train_df.to_csv('/content/drive/MyDrive/competitions/recog-r1/train.csv')
    test_df.to_csv('/content/drive/MyDrive/competitions/recog-r1/test.csv')
