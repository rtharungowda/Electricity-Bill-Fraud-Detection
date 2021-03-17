import pandas as pd

if __name__ == "__main__":
    df1 = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/train_invoice.csv')
    df2 = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/client_train.csv')
    df = pd.merge(df1,df2,on='client_id')
    # print(df.head())
    df.to_csv('/content/drive/MyDrive/competitions/recog-r1/train.csv')

    dft1 = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/test_invoice.csv')
    dft2 = pd.read_csv('/content/drive/MyDrive/competitions/recog-r1/client_test.csv')
    dft = pd.merge(df1,df2,on='client_id')
    dft.to_csv('/content/drive/MyDrive/competitions/recog-r1/test.csv')