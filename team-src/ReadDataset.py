from Imports import *


def readMergedDataset():
    # Reading in each data file
    transaction = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_transaction.csv')
    identity = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_identity.csv')

    # Merging transactional and test data
    merged_df = transaction.merge(identity, how='left', left_index=True, right_index=True)

    # Freeing up memory
    del transaction, identity

    return merged_df

def readMergedDatasetLocal(): 
    transaction = pd.read_csv("D:\School\Fifth Year\Large Scale Data Analytics\Project\ieee-fraud-detection\\train_transaction.csv")
    identity = pd.read_csv("D:\School\Fifth Year\Large Scale Data Analytics\Project\ieee-fraud-detection\\train_identity.csv")

    # Merging transactional and test data
    merged_df = transaction.merge(identity, how='left',on="TransactionID", left_index=True, right_index=True)

    # Freeing up memory
    del transaction, identity

    return merged_df

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