import pandas as pd


def readFiles():
    # Reading in each data file
    train_transaction = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_transaction.csv')
    train_identity = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_identity.csv')
    test_identity = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_identity.csv')
    test_transaction = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_transaction.csv')

    # Merging transactional and test data
    train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
    test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

    # Freeing up memory
    del train_identity, train_transaction, test_identity, test_transaction

    return train, test


# used for testing purposes
def readTrainTransaction():
    train_transaction = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_transaction.csv')
    return train_transaction


# used for testing purposes
def readTrainIdentity():
    train_identity = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_identity.csv')
    return train_identity


# used for testing purposes
def readTestTransaction():
    test_transaction = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_transaction.csv')
    return test_transaction


# used for testing purposes
def readTestIdentity():
    test_identity = pd.read_csv(
        '/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_identity.csv')
    return test_identity