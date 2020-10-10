import pandas as pd

train_transaction_ds = pd.read_csv("/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/train_transaction.csv")
train_identity_ds = pd.read_csv('/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/train_identity.csv')
test_identity_ds = pd.read_csv('/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/test_identity.csv')
test_transaction_ds = pd.read_csv('/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/test_transaction.csv')

df = pd.concat([train[col],test[col]],axis=0)
# PERFORM FEATURE ENGINEERING HERE
df[col].fillna(-999, inplace=True)

df[col],_ = df[col].factorize()

if df[col].max()<128: 
    df[col] = df[col].astype('int8')
elif df[col].max()<32768: 
    df[col] = df[col].astype('int16')
else: 
    df[col].astype('int32')

for col in df.columns:
    if df[col].dtype=='float64': df[col] = df[col].astype('float32')
    if df[col].dtype=='int64': df[col] = df[col].astype('int32')


train[col] = df[:len(train)]
test[col] = df[len(train):]