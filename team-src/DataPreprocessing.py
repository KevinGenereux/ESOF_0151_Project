from Imports import *


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

def get_missing_data_percentage(data):
    
    # where mvp = missing value percentages
    mvp = data.isnull().sum() * 100 / len(data)
    mvp = pd.DataFrame({'Feature': data.columns,'Percentage': mvp})
    
    return mvp.sort_values(by ='Percentage', ascending=False)

def drop_one_value_columns(data):
    
    # Drop columns with only 1 unique value.
    for column in data.columns:
        if len(data[column].unique()) == 1:
            #print(traindata[column].name)
            data.drop(column,inplace=True,axis=1)
            
    return data

def getCategoricalFeatures(data):
    columns = list(data)
    result = []
    for c in columns:
        if data.dtypes[c] == np.object:
            result.append(c)
    return data[result]

def getNumericalFeatures(data):
    columns = list(data)
    result = []
    for c in columns: 
        if data.dtypes[c] != np.object:
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
    return df

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

def split_data(features, labels):
    
    # Data Splitting: 60% for training, 20% for validation and 20% for testing.
    X_train, X_test, Y_train, y_test = train_test_split(features, labels, test_size=0.4)
    X_validation, X_test, Y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)
    
    return X_train, Y_train, X_test, y_test, X_validation, Y_validation

#Select the 50 best features 
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