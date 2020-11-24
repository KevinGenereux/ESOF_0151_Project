import numpy as np

def transformFeatures(df):
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    # New feature - decimal part of the transaction amount.
    df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)

    # New feature - day of week in which a transaction happened.
    df['Transaction_day_of_week'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7)

    # New feature - hour of the day in which a transaction happened.
    df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24

    return df

def generateNewFeatures(df):
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
    df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]

    df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
    df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

    df['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
    df['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]

    df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
    df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]

    df['id_34'] = df['id_34'].str.split(':', expand=True)[1]
    df['id_23'] = df['id_23'].str.split(':', expand=True)[1]

    df.drop(['DeviceInfo', 'id_30', 'id_31', 'id_33'], axis=1)

    return df

def dropStronglyCorrelatedFeatures(data, threshold):
    corr_matrix = data.corr()
    numCols = corr_matrix.shape[0]
    columns = np.full((numCols,), True, dtype=bool)

    for i in range(numCols):
        for j in range(i + 1, numCols):
            # if the
            if corr_matrix.iloc[i, j] >= threshold:
                    if columns[j]:
                        columns[j] = False

    selected_columns = data.columns[columns]
    data = data[selected_columns]
    return data

def selectkbestfeatures(X_train, Y_train, X_validation, X_test, numberOfFeatures):

    fit = SelectKBest(score_func=f_classif, k=numberOfFeatures).fit(X_train, Y_train)

    X_train = fit.transform(X_train)
    X_validation = fit.transform(X_validation)
    X_test = fit.transform(X_test)

    # Get column names from the best features
    X_train_cols = fit.get_support(indices=True)
    X_validation_cols = fit.get_support(indices=True)
    X_test_cols = fit.get_support(indices=True)

    X_train = pd.DataFrame(X_train, columns=X_train_cols)
    X_validation = pd.DataFrame(X_validation, columns=X_validation_cols)
    X_test = pd.DataFrame(X_test, columns=X_test_cols)

    # Create new dataframes with the column names
    #X_train = X_train.iloc[:,X_train_cols]
    #X_validation = X_validation.iloc[:,X_validation_cols]
    #X_test = X_test.iloc[:,X_test_cols]

    return X_train, X_validation, X_test