import pandas as pd

def readFiles():
    train_transaction = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_transaction.csv')
    train_identity = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_identity.csv')
    test_identity = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_identity.csv')
    test_transaction = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_transaction.csv')
    return train_transaction, train_identity, test_identity, test_transaction
