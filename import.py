import pandas as pd
import time

start_time = time.time()

train_transaction = pd.read_csv("/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/train_transaction.csv")
train_identity_ds = pd.read_csv('/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/train_identity.csv')
test_identity_ds = pd.read_csv('/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/test_identity.csv')
test_transaction_ds = pd.read_csv('/content/drive/My Drive/Large Scale Data Analytics Project/Project Code/dataset/test_transaction.csv')

print(test_identity_ds)
print("--- %s seconds ---" % (time.time() - start_time))