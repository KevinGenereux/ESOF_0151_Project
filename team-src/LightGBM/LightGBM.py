from ReadDataset import readMergedDataset
from DataPreprocessing import reduceMemUsage, encodeNumericalColumns, replaceMissingValues
from FeatureEngineering import transformFeatures, generateNewFeatures, dropStronglyCorrelatedFeatures
from TrainTestSplit import trainTestSplit
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Read in the dataset containing the merged transaction and identity tables
train = readMergedDataset()

# Reduce the memory size of the dataset, this will also help improve computational speed
train = reduceMemUsage(train)

y = train['isFraud']
train = train.drop(['isFraud', 'TransactionID_x'], axis=1)

train = encodeNumericalColumns(train)

# Replace dataset missing values
train = replaceMissingValues(train)

train = transformFeatures(train)
train = generateNewFeatures(train)
train = dropStronglyCorrelatedFeatures(train, 0.9)
print(train)

NFOLDS = 5
folds = KFold(n_splits=NFOLDS)

columns = train.columns
splits = folds.split(train, y)
y_oof = np.zeros(train.shape[0])

params = {'num_leaves': 300,
          'min_child_weight': 0.05,
          'feature_fraction': 0.5,
          'bagging_fraction': 0.5,
          'min_data_in_leaf': 100,
          'objective': 'binary',
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "metric": 'auc',
          "verbosity": -1,
          'random_state': 123,
          }

for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = train[columns].iloc[train_index], train[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, num_boost_round=100, valid_sets=[dtrain, dvalid], verbose_eval=200)

    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    del X_train, X_valid, y_train, y_valid
    gc.collect()

print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

