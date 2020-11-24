from Imports import *

def EvaluateModel(dtrain, dvalid, params):

    NFOLDS = 5
    folds = KFold(n_splits=NFOLDS)

    columns = X.columns
    splits = folds.split(X, y)
    y_preds = np.zeros(X_test.shape[0])
    y_oof = np.zeros(X.shape[0])
    score = 0

    feature_importances = pd.DataFrame()
    feature_importances['feature'] = columns

    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        clf = lgb.train(params, dtrain, 10000, valid_sets=[dtrain, dvalid], verbose_eval=200, early_stopping_rounds=20)

        feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

        y_pred_valid = clf.predict(X_valid)
        y_oof[valid_index] = y_pred_valid
        print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

        score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
        y_preds += clf.predict(X_test) / NFOLDS

        del X_train, X_valid, y_train, y_valid
        gc.collect()

    print(f"\nMean AUC = {score}")
    print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")