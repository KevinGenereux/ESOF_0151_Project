import numpy as np
from sklearn.preprocessing import LabelEncoder


def reduceMemUsage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            if str(col_type)[:3] == 'int':
                if df[col].max() < 2 ** 7:
                    df[col] = df[col].astype(np.int8)
                elif df[col].max() < 2 ** 15:
                    df[col] = df[col].astype(np.int16)
                elif df[col].max() < 2 ** 31:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if df[col].max() < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif df[col].max() < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def getCategoricalFeatures(data):
    columns = list(data)
    result = []
    for c in columns:
        if data.dtypes[c] == np.object:
            result.append(c)
    return data[result]


def replaceMissingValues(df):
    for col in df.columns:
        df[col].fillna(-9999, inplace=True)
        # fixes missing data by taking values from other rows and taking the average
        # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

        # this function takes the average of every column excluding the unknown values
        # imp.fit(data)

        # inserts the average into the missing spots
        # data = imp.fit_transform(data)

    return df

def encodeNumericalColumns(df):
    enc=LabelEncoder()# create the label encoder
    for col in df:#loop through all of the columns
        if df[col].dtype.name=="object":#if the data is in a string format we will need to convert it to numeric to find the correlation
            enc.fit(df[col].astype(str))#fit the column to the encoder to convert to numeric
            df[col]=enc.fit_transform(df[col].astype(str))#transform the data into numeric


def prepare_inputs_and_outputs(data):
    # Prepare & save the inputs and outputs features
    features = data.drop(['isFraud', 'TransactionID'], axis=1)
    labels = data[['isFraud']]

    return features, labels


def drop_high_missing_data_columns(mvd, data, threshold):
    # Where "mvd" = missing value data
    # Get names of indexes for which column missing data is over 50%
    high_missing_data_cols = mvd[mvd['Percentage'] > threshold].index

    for col_name in range(len(high_missing_data_cols)):
        del data[high_missing_data_cols[col_name]]  # Delete rows from dataFrame??? or columns

    return data


def drop_high_correlation_features(data, threshold):
    corr_matrix = data.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > threshold)]
    data = data.drop(columns=to_drop)

    return data

