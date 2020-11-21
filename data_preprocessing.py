import numpy
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




